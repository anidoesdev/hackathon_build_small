import json
import re
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
# ASR backend — Parakeet (NeMo) preferred, Whisper-small fallback
# ---------------------------------------------------------------------------
try:
    import nemo.collections.asr as nemo_asr  # noqa: F401
    ASR_BACKEND = "parakeet"
except Exception:
    ASR_BACKEND = "whisper"

print(f"[ASR] Using backend: {ASR_BACKEND}")

_asr_model = None  # lazy-loaded inside @spaces.GPU


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
            device=0 if _cuda_available() else -1,
        )
    return _asr_model


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@spaces.GPU
def transcribe(audio_path: str) -> str:
    if not audio_path:
        return "No audio provided — please record or upload a clip."
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
    # Remove ```json ... ``` or ``` ... ``` fences
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
        todos = _parse_json_list(raw)
        return todos
    except Exception as e:
        print(f"[extract_todos error] {e}")
        return []


# ---------------------------------------------------------------------------
# Conversational refinement (stub — wired fully in Task 4)
# ---------------------------------------------------------------------------

def refine_todos(current_list: list[dict], instruction: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_get_todos(audio, transcript_box, todo_state):
    transcript = transcribe(audio) if audio else "No audio provided."
    todos = extract_todos(transcript) if audio else []
    return transcript, render_todo_markdown(todos), todos


def handle_refine_text(instruction, chat_history, todo_state):
    if not instruction.strip():
        return chat_history, render_todo_markdown(todo_state), todo_state
    updated = refine_todos(todo_state, instruction)
    chat_history = chat_history + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "List updated."},
    ]
    return chat_history, render_todo_markdown(updated), updated


def handle_refine_voice(audio, chat_history, todo_state):
    if not audio:
        return chat_history, render_todo_markdown(todo_state), todo_state
    instruction = transcribe(audio)
    return handle_refine_text(instruction, chat_history, todo_state)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Meeting → To-Do Agent") as demo:
    todo_state = gr.State([])

    gr.Markdown("# 🎙️ Meeting → To-Do Agent")
    gr.Markdown(
        "Record yourself describing a meeting, click **Get to-dos**, "
        "then refine the list by typing or speaking."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Record your meeting")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Meeting audio",
            )
            get_todos_btn = gr.Button("Get to-dos", variant="primary")

            gr.Markdown("### 2 · Transcript")
            transcript_box = gr.Textbox(
                label="What was heard",
                lines=6,
                interactive=False,
                placeholder="Transcript will appear here…",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 3 · To-do list")
            todo_display = gr.Markdown(render_todo_markdown([]))

            gr.Markdown("### 4 · Refine the list")
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=200)

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

    # --- wire buttons ---
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


if __name__ == "__main__":
    demo.launch()
