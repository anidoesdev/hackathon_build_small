---
title: Meeting To-Do Agent
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
tags:
  - backyard-ai
  - openbmb
  - minicpm
  - audio
  - asr
  - meeting
  - productivity
  - gradio
---

# 🎙️ Meeting → To-Do Agent

> **Hugging Face Build Small Hackathon · Backyard AI track · Best MiniCPM Build (OpenBMB)**

## The problem

You just finished a messy, hour-long meeting. Everyone talked over each other, half the action items were buried in side conversations, and you still have three more calls today. You don't want to type a summary — you want to *talk through what happened* and get a clean list back instantly.

## What this does

1. **Record or upload** yourself describing the meeting (up to 10 minutes).
2. Click **Get to-dos** — the app transcribes your audio and extracts a structured action-item table (task, owner, deadline, priority).
3. **Keep talking** to refine the list: *"add a deadline of next Friday to item 2"*, *"drop the last one"*, *"who owns the budget task?"* — by voice or text.

The list updates live with every instruction. No forms. No templates. Just conversation.

## How to use

| Step | What to do |
|------|------------|
| 1 | Record yourself describing a meeting (microphone) or upload a WAV/MP3/M4A file |
| 2 | Click **Get to-dos** and wait ~30 s for transcription + extraction |
| 3 | Read the transcript to sanity-check what was heard |
| 4 | Type or speak refinement instructions in the chat box |
| 5 | Click **Export to Markdown** to download or copy the final list |

## Tech stack

| Component | Model / Library |
|-----------|----------------|
| Speech-to-text | NVIDIA Parakeet `nvidia/parakeet-tdt-0.6b-v2` via NeMo *(fallback: `openai/whisper-small`)* |
| Extraction + refinement LLM | MiniCPM4.1-8B `openbmb/MiniCPM4.1-8B`, 4-bit NF4 via bitsandbytes |
| UI | Gradio 5.x, deployed as a ZeroGPU Space |

**Model size note:** the 8B LLM is a deliberate accuracy tradeoff — it handles messy transcripts and nuanced refinement instructions far better than a 1–4B model. This means we are not eligible for the Tiny Titan badge but ARE eligible for **Backyard AI** and **Best MiniCPM Build**.

## Architecture

```
[microphone / upload]
        │
        ▼
   @spaces.GPU
  transcribe()          ← Parakeet (NeMo) or Whisper-small
        │
        ▼
   @spaces.GPU
 extract_todos()        ← MiniCPM4.1-8B  +  EXTRACT_PROMPT
        │
        ▼
   gr.State (list[dict])
        │
   ┌────┴────┐
   │         │
  text      voice
  refine    refine
        │
        ▼
   @spaces.GPU
  refine_todos()        ← MiniCPM4.1-8B  +  REFINE_PROMPT
```

Both `transcribe` and LLM calls are lazy-loaded singletons inside `@spaces.GPU` functions, so model weights are only moved to GPU when a button is pressed — ZeroGPU compatible.

## Deploying to a Hugging Face Space

### Requirements

- Space type: **ZeroGPU** (A100 40 GB)
- SDK: `gradio` (set in frontmatter above)
- `app_file: app.py`
- Org: `build-small-hackathon` (or your own)

### Deploy checklist

- [ ] Push this repo to a Space in the `build-small-hackathon` org
- [ ] Set hardware to **ZeroGPU** in Space settings
- [ ] Confirm `sdk: gradio` and `app_file: app.py` are correct in this README's frontmatter
- [ ] First cold-start will download ~5 GB of model weights; subsequent requests use the cache
- [ ] (Optional) Add `sample_meeting.wav` to the repo root for the example clip

### `requirements.txt`

```
gradio>=5.0.0,<6.0.0
transformers>=4.45.0
torch>=2.4.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
spaces>=0.30.0
nemo_toolkit[asr]>=2.0.0
```

## Demo

<!-- TODO: add demo video link after recording -->

## Social post

<!-- TODO: add social post link -->

---

*Built for the [Hugging Face Build Small hackathon](https://huggingface.co/build-small) — Backyard AI track.*
