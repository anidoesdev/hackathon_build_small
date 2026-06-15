import contextlib
import json
import os
import re
import tempfile
import wave
import gradio as gr

# ---------------------------------------------------------------------------
# spaces — no-op shim when running outside ZeroGPU
# ---------------------------------------------------------------------------
try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn

# ---------------------------------------------------------------------------
# ASR — Parakeet (NeMo) preferred, Whisper-small fallback
# ---------------------------------------------------------------------------
try:
    import nemo.collections.asr as nemo_asr  # noqa: F401
    ASR_BACKEND = "parakeet"
except Exception:
    ASR_BACKEND = "whisper"

print(f"[ASR] Using backend: {ASR_BACKEND}")

_asr_model = None

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}
MAX_AUDIO_SECONDS = 600       # 10 minutes — guard against ZeroGPU timeout
MAX_TRANSCRIPT_WORDS = 2500   # prevent LLM context overflow on long recordings

PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}
PRIORITY_CLEAN = {
    "🔴 high": "high", "🟡 medium": "medium", "🟢 low": "low",
    "high": "high", "medium": "medium", "low": "low",
}

ERROR_PREFIXES = (
    "[Transcription error",
    "[Audio too long",
    "Unsupported file format",
    "No audio provided",
    "Could not detect speech",
)

DF_HEADERS = ["Task", "Owner", "Due", "Priority"]


def _is_asr_error(text: str) -> bool:
    return any(text.startswith(p) for p in ERROR_PREFIXES)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _audio_duration(path: str) -> float | None:
    """Return duration in seconds for any audio format via mutagen; WAV fallback."""
    try:
        from mutagen import File as MutaFile
        f = MutaFile(path)
        if f is not None and f.info is not None:
            return f.info.length
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(path, "r")) as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass
    return None


def _load_asr():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    if ASR_BACKEND == "parakeet":
        import nemo.collections.asr as nemo_asr
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2"
        )
        _asr_model.eval()
    else:
        from transformers import pipeline
        _asr_model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            stride_length_s=5,
            device=0 if _cuda_available() else -1,
        )
    return _asr_model


@spaces.GPU
def transcribe(audio_path: str) -> str:
    if not audio_path:
        return "No audio provided — please record or upload a clip."

    ext = os.path.splitext(audio_path)[1].lower()
    if ext and ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return (
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    duration = _audio_duration(audio_path)
    if duration is not None and duration > MAX_AUDIO_SECONDS:
        return (
            f"[Audio too long: {duration:.0f} s — please keep recordings under "
            f"{MAX_AUDIO_SECONDS // 60} minutes]"
        )

    model = _load_asr()
    try:
        if ASR_BACKEND == "parakeet":
            results = model.transcribe([audio_path])
            text = results[0] if isinstance(results[0], str) else results[0].text
        else:
            out = model(audio_path, return_timestamps=False)
            text = out["text"].strip()
    except Exception as e:
        return f"[Transcription error: {e}]"

    return text.strip() if text.strip() else "Could not detect speech in the audio."


# ---------------------------------------------------------------------------
# LLM — MiniCPM4.1-8B, 4-bit NF4
# ---------------------------------------------------------------------------

_llm_model = None
_llm_tokenizer = None

LLM_MODEL_ID = "openbmb/MiniCPM4.1-8B"

EXTRACT_PROMPT = """\
You are a meeting assistant. The user gives you a raw, possibly messy
transcript of someone describing a meeting out loud. Extract a clear,
deduplicated list of action items (to-dos).

For each action item capture:
- task: short imperative description of what needs to be done
- owner: who is responsible ("unassigned" if not stated)
- due: a deadline if mentioned, else null
- priority: "high" | "medium" | "low" (infer from urgency; default "medium")

Rules:
- Only concrete, actionable tasks. Ignore discussion, opinions, FYI statements.
- Merge duplicates. Do not invent tasks not supported by the transcript.
- Output ONLY a valid JSON array of objects with keys
  task, owner, due, priority. No markdown, no commentary."""

REFINE_PROMPT = """\
You are a meeting assistant maintaining a to-do list. You receive:
1. the current to-do list as a JSON array
2. a user instruction to modify it (add / remove / edit / reprioritize /
   assign an owner / set a deadline)

Apply ONLY the requested change; keep every other item unchanged.
Output ONLY the full updated JSON array, same schema
(task, owner, due, priority). No commentary, no markdown."""


def _load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _llm_model.eval()
    return _llm_model, _llm_tokenizer


def _llm_generate(messages: list[dict], max_new_tokens: int = 1024) -> str:
    import torch
    model, tokenizer = _load_llm()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_json_list(raw: str) -> list[dict]:
    """Robustly extract and parse a JSON array from LLM output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # Slice from first '[' to last ']' to skip any preamble text
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def _truncate_transcript(text: str) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= MAX_TRANSCRIPT_WORDS:
        return text, False
    return " ".join(words[:MAX_TRANSCRIPT_WORDS]), True


@spaces.GPU
def extract_todos(transcript: str) -> list[dict]:
    if not transcript or _is_asr_error(transcript):
        return []
    transcript, truncated = _truncate_transcript(transcript)
    if truncated:
        print(f"[extract_todos] Transcript truncated to {MAX_TRANSCRIPT_WORDS} words")
    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]
    try:
        raw = _llm_generate(messages)
        return _parse_json_list(raw)
    except Exception as e:
        print(f"[extract_todos error] {e}")
        return []


@spaces.GPU
def refine_todos(current_list: list[dict], instruction: str) -> list[dict]:
    # Allow refinement even on an empty list so users can build from scratch via chat
    messages = [
        {"role": "system", "content": REFINE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current list:\n{json.dumps(current_list, indent=2)}"
                f"\n\nInstruction: {instruction}"
            ),
        },
    ]
    try:
        raw = _llm_generate(messages)
        updated = _parse_json_list(raw)
        if isinstance(updated, list):
            return updated
        print(f"[refine_todos] parse failed, keeping previous. Raw: {raw[:300]}")
        return current_list
    except Exception as e:
        print(f"[refine_todos error] {e}")
        return current_list


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def todos_to_df(todo_list: list[dict]) -> list[list]:
    rows = []
    for item in todo_list:
        p = item.get("priority", "medium")
        emoji = PRIORITY_EMOJI.get(p, "🟡")
        rows.append([
            item.get("task", ""),
            item.get("owner", "unassigned"),
            item.get("due") or "",
            f"{emoji} {p}",
        ])
    return rows


def df_to_todos(df_data) -> list[dict]:
    if df_data is None:
        return []
    try:
        rows = df_data.values.tolist()
    except AttributeError:
        rows = list(df_data)
    result = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        task = str(row[0]).strip()
        owner = str(row[1]).strip() if len(row) > 1 and row[1] else "unassigned"
        due_raw = str(row[2]).strip() if len(row) > 2 and row[2] else ""
        raw_p = str(row[3]).strip() if len(row) > 3 and row[3] else "medium"
        result.append({
            "task": task,
            "owner": owner or "unassigned",
            "due": due_raw or None,
            "priority": PRIORITY_CLEAN.get(raw_p, "medium"),
        })
    return result


def _csv_escape(s: str) -> str:
    return '"' + str(s).replace('"', '""') + '"'


def export_todos_markdown(todo_list: list[dict]) -> str:
    if not todo_list:
        return "# Meeting To-Dos\n\n_No items._"
    lines = ["# Meeting To-Dos\n"]
    for item in todo_list:
        due = f", due {item['due']}" if item.get("due") else ""
        owner = item.get("owner", "unassigned")
        priority = item.get("priority", "medium")
        emoji = PRIORITY_EMOJI.get(priority, "🟡")
        lines.append(
            f"- [ ] **{item.get('task', '')}** — {owner}{due} _{emoji} {priority} priority_"
        )
    return "\n".join(lines)


def export_todos_csv(todo_list: list[dict]) -> str:
    lines = ["task,owner,due,priority"]
    for item in todo_list:
        lines.append(",".join([
            _csv_escape(item.get("task", "")),
            _csv_escape(item.get("owner", "unassigned")),
            _csv_escape(item.get("due") or ""),
            _csv_escape(item.get("priority", "medium")),
        ]))
    return "\n".join(lines)


def handle_export(todo_state):
    md = export_todos_markdown(todo_state)
    csv_text = export_todos_csv(todo_state)

    tmp_md = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8", prefix="meeting_todos_"
    )
    tmp_md.write(md)
    tmp_md.close()

    tmp_csv = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", prefix="meeting_todos_"
    )
    tmp_csv.write(csv_text)
    tmp_csv.close()

    return md, tmp_md.name, tmp_csv.name


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_get_todos(audio, transcript_box, todo_state, progress=gr.Progress(track_tqdm=False)):
    if not audio:
        yield "No audio provided — please record or upload a clip.", todos_to_df([]), [], "_No to-dos yet._"
        return

    progress(0.1, desc="Transcribing audio…")
    yield "Transcribing…", todos_to_df(todo_state), todo_state, "_Transcribing audio, please wait…_"

    transcript = transcribe(audio)

    if _is_asr_error(transcript):
        yield transcript, todos_to_df([]), [], "_Could not process audio — see transcript box._"
        return

    progress(0.5, desc="Extracting to-dos…")
    yield transcript, todos_to_df(todo_state), todo_state, "_Extracting action items…_"

    todos = extract_todos(transcript)

    progress(1.0, desc="Done")
    yield transcript, todos_to_df(todos), todos, f"_✅ {len(todos)} action item(s) extracted._"


def handle_reextract(transcript, todo_state):
    if not transcript or _is_asr_error(transcript):
        return todos_to_df(todo_state), todo_state, "_No valid transcript to extract from._"
    todos = extract_todos(transcript)
    return todos_to_df(todos), todos, f"_✅ Re-extracted {len(todos)} action item(s)._"


def handle_refine_text(instruction, chat_history, todo_state):
    instruction = instruction.strip()
    if not instruction:
        yield chat_history, todos_to_df(todo_state), todo_state
        return

    pending_history = chat_history + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "…"},
    ]
    yield pending_history, todos_to_df(todo_state), todo_state

    updated = refine_todos(todo_state, instruction)

    changed = updated != todo_state
    reply = "Done — list updated." if changed else "No changes needed."
    final_history = chat_history + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": reply},
    ]
    yield final_history, todos_to_df(updated), updated


def handle_refine_voice(audio, chat_history, todo_state):
    if not audio:
        yield chat_history, todos_to_df(todo_state), todo_state
        return

    pending_history = chat_history + [{"role": "assistant", "content": "Transcribing your instruction…"}]
    yield pending_history, todos_to_df(todo_state), todo_state

    instruction = transcribe(audio)

    if _is_asr_error(instruction):
        error_history = chat_history + [
            {"role": "assistant", "content": f"Sorry, transcription failed: {instruction}"}
        ]
        yield error_history, todos_to_df(todo_state), todo_state
        return

    yield from handle_refine_text(instruction, chat_history, todo_state)


def handle_df_edit(df_data):
    return df_to_todos(df_data)


def handle_reset():
    return "", todos_to_df([]), [], [], "_No to-dos yet. Record or upload audio and click **Get to-dos**._"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

EXAMPLE_AUDIO = "sample_meeting.wav"

with gr.Blocks(title="Meeting → To-Do Agent") as demo:
    todo_state = gr.State([])

    gr.Markdown("# 🎙️ Meeting → To-Do Agent")
    gr.Markdown(
        "**Record yourself describing a meeting → get a clean to-do list → "
        "keep talking to refine it.** Record or upload up to 10 minutes of audio, "
        "then use voice or text to add deadlines, change owners, or reprioritise."
    )
    gr.Markdown(
        "> ⏳ **First run may take 60–90 s** while model weights load into GPU memory. "
        "Subsequent requests are much faster."
    )

    with gr.Row():
        # ── Left column: audio input + transcript ──────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Record your meeting")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Meeting audio (WAV / MP3 / M4A / …, max 10 min)",
            )
            get_todos_btn = gr.Button("Get to-dos", variant="primary")

            if os.path.exists(EXAMPLE_AUDIO):
                gr.Examples(
                    examples=[[EXAMPLE_AUDIO]],
                    inputs=[audio_input],
                    label="Try the example clip",
                )

            gr.Markdown("### 2 · Transcript")
            transcript_box = gr.Textbox(
                label="What was heard (editable — fix errors then click Re-extract)",
                lines=6,
                interactive=True,
                placeholder="Transcript will appear here…",
            )
            reextract_btn = gr.Button("Re-extract from transcript", variant="secondary")

        # ── Right column: to-do list + refinement ─────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 3 · To-do list")
            todo_status = gr.Markdown(
                "_No to-dos yet. Record or upload audio and click **Get to-dos**._"
            )
            todo_df = gr.Dataframe(
                headers=DF_HEADERS,
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                interactive=True,
                label="Action items (click any cell to edit directly)",
                row_count=(0, "dynamic"),
            )

            gr.Markdown("### 4 · Refine the list")
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=220,
                allow_tags=False,
            )

            with gr.Row():
                refine_text = gr.Textbox(
                    placeholder='e.g. "add deadline next Friday to item 1"',
                    label="Type an instruction",
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1)

            refine_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Or speak your instruction",
            )

            reset_btn = gr.Button("🔄 Start over", variant="stop", size="sm")

            with gr.Accordion("Export", open=False):
                export_btn = gr.Button("Generate export")
                export_md_box = gr.Textbox(
                    label="Markdown (copy-paste)",
                    lines=6,
                    show_copy_button=True,
                    interactive=False,
                )
                with gr.Row():
                    export_md_file = gr.File(label="Download .md")
                    export_csv_file = gr.File(label="Download .csv")

    # ── Wire events ─────────────────────────────────────────────────────────
    get_todos_btn.click(
        handle_get_todos,
        inputs=[audio_input, transcript_box, todo_state],
        outputs=[transcript_box, todo_df, todo_state, todo_status],
    )

    reextract_btn.click(
        handle_reextract,
        inputs=[transcript_box, todo_state],
        outputs=[todo_df, todo_state, todo_status],
    )

    todo_df.change(
        handle_df_edit,
        inputs=[todo_df],
        outputs=[todo_state],
    )

    send_btn.click(
        handle_refine_text,
        inputs=[refine_text, chatbot, todo_state],
        outputs=[chatbot, todo_df, todo_state],
    ).then(lambda: "", outputs=refine_text)

    refine_text.submit(
        handle_refine_text,
        inputs=[refine_text, chatbot, todo_state],
        outputs=[chatbot, todo_df, todo_state],
    ).then(lambda: "", outputs=refine_text)

    refine_audio.change(
        handle_refine_voice,
        inputs=[refine_audio, chatbot, todo_state],
        outputs=[chatbot, todo_df, todo_state],
    )

    reset_btn.click(
        handle_reset,
        outputs=[transcript_box, todo_df, todo_state, chatbot, todo_status],
    )

    export_btn.click(
        handle_export,
        inputs=[todo_state],
        outputs=[export_md_box, export_md_file, export_csv_file],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
