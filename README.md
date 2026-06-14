---
title: Meeting To-Do Agent
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
tags:
  - backyard-ai
  - openbmb
  - minicpm
  - asr
  - audio
  - meeting
  - productivity
---

# Meeting → To-Do Voice Agent

**Hugging Face Build Small Hackathon · Backyard AI track**

Talk through your meeting out loud — get back a clean, actionable to-do list. Then keep talking to refine it.

## How to use

1. Record (or upload) yourself describing a meeting.
2. Click **Get to-dos** — the app transcribes your audio and extracts action items.
3. Refine the list by typing (or recording) instructions like "add a deadline to item 2" or "drop the last one".

## Tech stack

- **ASR:** NVIDIA Parakeet `nvidia/parakeet-tdt-0.6b-v2` via NeMo (fallback: `openai/whisper-small`)
- **LLM:** MiniCPM4.1-8B (`openbmb/MiniCPM4.1-8B`) in 4-bit NF4 via bitsandbytes
- **UI:** Gradio (ZeroGPU Space)

## Track tags

- Backyard AI
- Best MiniCPM Build (OpenBMB)

## Demo

<!-- TODO: add demo video link -->

## Social post

<!-- TODO: add social post link -->
