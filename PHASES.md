# Phases for Message Notification Router

This document outlines the step-by-step phases to build the AI-powered Message Notification Router.

## Phase 1: Setup & Data Ingestion
- Set up a Python environment and install core dependencies (`pandas`, `openai` / `anthropic` / `google-genai`, `python-dotenv`).
- Load environment variables securely.
- Write a data loader module to read all provided `.csv` files from `dataset/` into Pandas DataFrames.
- Implement a helper function to extract the complete, unified context for any given `message_id` (joining user preferences, group info, business history, and message history).

## Phase 2: Core Routing Logic (Text-Only)
- Design the initial System Prompt defining the routing rules (`notify`, `digest`, `mute`) and available `message_type` categories.
- Create an API wrapper function that calls the LLM with structured outputs (JSON Schema) to guarantee the response matches the required output columns: `action`, `message_type`, `reason`, `confidence`, and `evidence_message_ids`.
- Run initial tests on text-only sample messages from `dataset/sample_messages.csv` to validate and tweak the prompt.

## Phase 3: Multimodal Processing (Images & Audio)
- **Image Handling**: If `media_type` is `image`, retrieve the file path from `dataset/images.csv`, read the local image file, and pass it directly to the vision-capable LLM alongside the prompt.
- **Audio Handling**: If `media_type` is `voice`, retrieve the file path from `dataset/voice_notes.csv`, transcribe the audio using a speech-to-text API (e.g., Whisper), and inject the transcription as text into the prompt.
- Test the multimodal capabilities on sample multimedia messages.

## Phase 4: Batch Processing & Generation
- Build a robust runner script to iterate over all rows in `dataset/messages.csv`.
- Add error handling, retry logic, and rate-limiting (to respect API limits).
- Write predictions incrementally or in bulk to generate the final `output.csv`.

## Phase 5: Evaluation & Submission Preparation
- Evaluate the outputs to ensure reasonable confidence calibration and correct identification of edge cases (e.g., scams, spam).
- Verify that `output.csv` has the exact required columns and row counts.
- Finalize code structure and document setup/run instructions in a submission `README.md`.
- Package the final `code.zip`, `output.csv`, and `log.txt` (chat transcript) for HackerRank submission.
