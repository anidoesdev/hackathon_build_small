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
MAX_AUDIO_SECONDS = 600  # 10 minutes — guard against ZeroGPU timeout


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _wav_duration(path: str) -> float | None:
    """Return duration in seconds for WAV files; None for other formats."""
    if not path.lower().endswith(".wav"):
        return None
    try:
        with contextlib.closing(wave.open(path, "r")) as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
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
            chunk_length_s=30,   # handles recordings longer than 30 s
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

    duration = _wav_duration(audio_path)
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
    """Strip code fences and parse a JSON array; returns [] on any failure."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


@spaces.GPU
def extract_todos(transcript: str) -> list[dict]:
    if not transcript or transcript.startswith("["):
        return []
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
    if not current_list:
        return current_list
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
        if updated:
            return updated
        print(f"[refine_todos] parse failed, keeping previous. Raw: {raw[:300]}")
        return current_list
    except Exception as e:
        print(f"[refine_todos error] {e}")
        return current_list


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_todo_markdown(todo_list: list[dict]) -> str:
    if not todo_list:
        return "_No to-dos yet. Record or upload audio and click **Get to-dos**._"
    lines = [
        "| # | Task | Owner | Due | Priority |",
        "|---|------|-------|-----|----------|",
    ]
    for i, item in enumerate(todo_list, 1):
        lines.append(
            f"| {i} | {item.get('task', '')} | {item.get('owner', 'unassigned')} "
            f"| {item.get('due') or '—'} | {item.get('priority', 'medium')} |"
        )
    return "\n".join(lines)


def export_todos_markdown(todo_list: list[dict]) -> str:
    if not todo_list:
        return "# Meeting To-Dos\n\n_No items._"
    lines = ["# Meeting To-Dos\n"]
    for item in todo_list:
        due = f", due {item['due']}" if item.get("due") else ""
        owner = item.get("owner", "unassigned")
        priority = item.get("priority", "medium")
        lines.append(f"- [ ] **{item.get('task', '')}** — {owner}{due} _{priority} priority_")
    return "\n".join(lines)


def handle_export(todo_state):
    md = export_todos_markdown(todo_state)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8", prefix="meeting_todos_"
    )
    tmp.write(md)
    tmp.close()
    return md, tmp.name


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_get_todos(audio, transcript_box, todo_state, progress=gr.Progress(track_tqdm=False)):
    if not audio:
        yield "No audio provided — please record or upload a clip.", render_todo_markdown([]), []
        return

    progress(0.1, desc="Transcribing audio…")
    yield "Transcribing…", "_Transcribing audio, please wait…_", todo_state

    transcript = transcribe(audio)

    if transcript.startswith("[Transcription error") or transcript.startswith("Unsupported") or transcript.startswith("[Audio too long"):
        yield transcript, "_Could not process audio — see transcript box for details._", []
        return

    progress(0.5, desc="Extracting to-dos…")
    yield transcript, "_Extracting action items…_", todo_state

    todos = extract_todos(transcript)

    progress(1.0, desc="Done")
    yield transcript, render_todo_markdown(todos), todos


def handle_refine_text(instruction, chat_history, todo_state):
    instruction = instruction.strip()
    if not instruction:
        yield chat_history, render_todo_markdown(todo_state), todo_state
        return

    pending_history = chat_history + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "…"},
    ]
    yield pending_history, render_todo_markdown(todo_state), todo_state

    updated = refine_todos(todo_state, instruction)

    changed = updated != todo_state
    reply = "Done — list updated." if changed else "No changes needed."
    final_history = chat_history + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": reply},
    ]
    yield final_history, render_todo_markdown(updated), updated


def handle_refine_voice(audio, chat_history, todo_state):
    if not audio:
        yield chat_history, render_todo_markdown(todo_state), todo_state
        return

    pending_history = chat_history + [{"role": "assistant", "content": "Transcribing your instruction…"}]
    yield pending_history, render_todo_markdown(todo_state), todo_state

    instruction = transcribe(audio)

    if instruction.startswith("[Transcription error"):
        error_history = chat_history + [{"role": "assistant", "content": f"Sorry, transcription failed: {instruction}"}]
        yield error_history, render_todo_markdown(todo_state), todo_state
        return

    yield from handle_refine_text(instruction, chat_history, todo_state)


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
                label="What was heard",
                lines=6,
                interactive=False,
                placeholder="Transcript will appear here…",
            )

        # ── Right column: to-do list + refinement ─────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 3 · To-do list")
            todo_display = gr.Markdown(render_todo_markdown([]))

            gr.Markdown("### 4 · Refine the list")
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=200,
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

            with gr.Accordion("Export to Markdown", open=False):
                export_btn = gr.Button("Generate export")
                export_md_box = gr.Textbox(
                    label="Markdown (copy-paste)",
                    lines=6,
                    show_copy_button=True,
                    interactive=False,
                )
                export_file = gr.File(label="Download .md file")

    # ── Wire events ─────────────────────────────────────────────────────────
    get_todos_btn.click(
        handle_get_todos,
        inputs=[audio_input, transcript_box, todo_state],
        outputs=[transcript_box, todo_display, todo_state],
    )

    send_btn.click(
        handle_refine_text,
        inputs=[refine_text, chatbot, todo_state],
        outputs=[chatbot, todo_display, todo_state],
    ).then(lambda: "", outputs=refine_text)

    refine_text.submit(
        handle_refine_text,
        inputs=[refine_text, chatbot, todo_state],
        outputs=[chatbot, todo_display, todo_state],
    ).then(lambda: "", outputs=refine_text)

    refine_audio.change(
        handle_refine_voice,
        inputs=[refine_audio, chatbot, todo_state],
        outputs=[chatbot, todo_display, todo_state],
    )

    export_btn.click(
        handle_export,
        inputs=[todo_state],
        outputs=[export_md_box, export_file],
    )


if __name__ == "__main__":
    demo.launch()
