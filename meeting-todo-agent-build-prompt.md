# Build Prompt — Conversational "Meeting → To-Do" voice agent
### Hugging Face Build Small · Backyard AI track

> Paste this whole file into your coding assistant (Codex / Claude Code / Cursor).
> Tell it: **"Work one task at a time. After each task, stop, show me what you built and how to test it, and wait for my OK before continuing."**

---

## Context for the assistant

I'm building a small AI app for the Hugging Face **Build Small** hackathon (Backyard AI track). The problem: a friend wants to **talk through a meeting out loud** and get back a clean list of to-dos — and then **keep talking to the app to refine that list** ("add a deadline to item 2", "drop the last one", "who owns the budget task?").

**Hard constraints:**
- Open-weight models only, **≤ 32B params**.
- Must be a **Gradio app** (deployed as a Gradio Space). No separate web frontend.
- Must run on Hugging Face **ZeroGPU** — decorate inference functions with `@spaces.GPU`.
- Minimal, pinned `requirements.txt`.

**Tech stack:**
- ASR (speech-to-text): NVIDIA **Parakeet** (`nvidia/parakeet-tdt-0.6b-v2`, English) via NeMo.
  - **Fallback:** if NeMo install/ZeroGPU gives trouble for more than ~1 hour, switch to `openai/whisper-small` via a `transformers` pipeline and keep moving. Note which one you used.
- Extraction + refinement LLM: **MiniCPM4.1-8B** (`openbmb/MiniCPM4.1-8B`), loaded in **4-bit** via bitsandbytes (NF4) to fit ZeroGPU comfortably and run fast.
- UI: `gradio`.

**Prize note:** the 8B LLM means we are NOT eligible for the Tiny Titan badge (that needs all models ≤4B) — that's a deliberate accuracy tradeoff. We ARE eligible for Backyard AI + OpenBMB's Best MiniCPM Build.

**Pipeline:** record audio → Parakeet transcribes → MiniCPM extracts a structured to-do list → user refines the list conversationally (voice or text) → list updates live.

Work through the tasks **in order**. Stop after each for review.

---

## Task 0 — Scaffold + empty conversational UI

**Do:**
- Create `app.py`, `requirements.txt`, `README.md`.
- `requirements.txt` pins: `gradio`, `transformers`, `torch`, `accelerate`, `bitsandbytes`, `spaces`, and `nemo_toolkit[asr]` (comment a `# fallback: remove nemo, model = whisper-small` line next to it).
- `app.py` — Gradio Blocks UI with:
  - audio input `gr.Audio(sources=["microphone","upload"], type="filepath")` + a "Get to-dos" button,
  - a transcript textbox,
  - a to-do list display (Markdown or `gr.Dataframe`),
  - a **chat box** for refinements (text input + send; we'll add voice refinement later),
  - a `gr.State` to hold the current to-do list (list of dicts).
- `README.md`: HF Space frontmatter (`sdk: gradio`, `app_file: app.py`) + placeholders for track tags.

**Review checkpoint — show me:** file tree, full `app.py`, and confirmation `python app.py` renders the full UI (all controls present, nothing wired). **Acceptance:** launches with no errors.

---

## Task 1 — Speech-to-text (Parakeet, with Whisper fallback)

**Do:**
- `transcribe(audio_path) -> str` using `nvidia/parakeet-tdt-0.6b-v2`. Load the model once at module level; run inference inside an `@spaces.GPU` function.
- Handle None/empty audio gracefully (friendly message, no crash).
- If NeMo blocks you for ~1 hour, implement the `transformers` `whisper-small` fallback instead and say so.

**Review checkpoint — show me:** the function, how the model loads/caches, and how to test (record ~20s describing a fake meeting → transcript appears). **Acceptance:** real clip → accurate transcript; empty input → message not error. Tell me which ASR you ended up using.

---

## Task 2 — To-do extraction (MiniCPM4.1-8B, 4-bit)

**Do:**
- Load `openbmb/MiniCPM4.1-8B` with `BitsAndBytesConfig` (4-bit NF4, `bnb_4bit_compute_dtype=torch.bfloat16`). Load once; run in an `@spaces.GPU` function.
- `extract_todos(transcript) -> list[dict]` using the **exact** prompt below. Force JSON-only; parse robustly (strip code fences, `try/except` on `json.loads`, fall back to `[]` + error note rather than crashing).
- Each item has keys: `task`, `owner`, `due`, `priority`.

Extraction prompt (verbatim):
```
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
  task, owner, due, priority. No markdown, no commentary.
```

**Review checkpoint — show me:** the function + JSON parse/fallback logic, and a test where I paste a transcript and get a clean list. **Acceptance:** valid transcript → well-formed list; malformed model output doesn't crash.

---

## Task 3 — Wire the first pass (audio → list)

**Do:**
- Button: audio → `transcribe` → `extract_todos` → store result in `gr.State` and render the list (checklist/table).
- Show the transcript too, so the user can sanity-check what was heard.
- Loading/progress states.

**Review checkpoint — show me:** a full run (speak → transcript → to-do list). **Acceptance:** end-to-end first pass works from one button press; the list is held in state.

---

## Task 4 — Conversational refinement (core feature)

**Do:**
- `refine_todos(current_list, instruction) -> list[dict]`: send the current list + the user's instruction to MiniCPM with the **refine prompt** below; replace `gr.State` with the returned list and re-render.
- Wire the chat box: typed instruction → `refine_todos` → list updates; echo a short confirmation in the chat.
- **Voice refinement:** also let the user record a short instruction; route it through `transcribe` then into `refine_todos`, so they can talk to it the same way.

Refine prompt (verbatim):
```
You are a meeting assistant maintaining a to-do list. You receive:
1. the current to-do list as a JSON array
2. a user instruction to modify it (add / remove / edit / reprioritize /
   assign an owner / set a deadline)

Apply ONLY the requested change; keep every other item unchanged.
Output ONLY the full updated JSON array, same schema
(task, owner, due, priority). No commentary, no markdown.
```

**Review checkpoint — show me:** I add/remove/edit items by typing AND by voice, and the rendered list updates correctly each time without losing other items. **Acceptance:** multi-turn refinement works; state persists across turns.

---

## Task 5 — Polish + deploy + submit

**Do:**
- Add an example audio clip + a one-line "how to use".
- Robustness: chunk/limit long audio; clear errors for unsupported files; guard against the LLM returning broken JSON mid-conversation (keep the previous list if a refine fails).
- Add an "export to .md / copy" button for the final list.
- Finalize `README.md`: idea write-up, tech stack, **frontmatter track tags** (Backyard AI + the MiniCPM/OpenBMB tag), and a link to the social post.
- Deploy notes for an HF Space in the `build-small-hackathon` org: ZeroGPU hardware, `sdk: gradio`, correct `app_file`.

**Review checkpoint — show me:** a deploy checklist + final README. **Acceptance:** Space runs live; README complete. I still owe: demo video + social post.

---

## Submission reminders (I do these, not the assistant)
- [ ] Gradio Space deployed inside the official Build Small org
- [ ] Demo video recorded (judges use it even if a live run hits GPU limits)
- [ ] One social-media post, linked from the Space README
- [ ] Frontmatter tags set (Backyard AI + Best MiniCPM Build)
