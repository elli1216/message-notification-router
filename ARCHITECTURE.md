# Message Notification Router — Architecture & Tradeoffs

Final rundown of the system built for the HackerRank Orchestrate challenge.

## Architecture

```
dataset/*.csv ──► data_loader.py ──► indexed DataFrames (O(1) lookups)
                      │
messages.csv row ──► router.py
                      ├── get_message_context()  → user prefs, group info, membership,
                      │                           business info + history
                      ├── get_evidence()         → top-3 relevant past messages
                      │                           + user's actual reaction (opened/replied/
                      │                           dismissed/muted/reported)
                      ├── build_prompt()         → routing rules + context + evidence
                      │                           + media bytes (image/audio)
                      └── Gemini (temp 0)        → JSON schema-validated decision
                      │
batch_processor.py ──► output.csv written after every message (resumable)
                      └── 4.5s throttle · 60s 429 backoff · abort on quota exhaustion
```

**Flow**: each of the 110 incoming messages gets a personalized context bundle →
heuristic retrieval of the user's most relevant history → one deterministic LLM
call (text, image, or audio attached) → structured
`action / message_type / reason / confidence / evidence_message_ids` →
incrementally persisted CSV. `evaluation/main.py` scores the same path against
the 21 solved samples.

## Components

| File | Role |
|---|---|
| `code/data_loader.py` | Loads all CSVs from `dataset/` into DataFrames with indices for O(1) lookups |
| `code/router.py` | Routing core: context assembly, evidence retrieval, prompt construction, multimodal media handling, structured-output LLM call |
| `code/batch_processor.py` | Batch runner: incremental saves, resume support, rate-limit handling, `--reset` for full reruns |
| `code/evaluation/main.py` | Scores routing accuracy against the solved `sample_messages.csv` |

## Tradeoffs

| Decision | Chose | Gave up | Rationale |
|---|---|---|---|
| **Routing engine** | Pure LLM, temp 0, JSON schema | Hand-written rule set / hybrid | ~15 interacting signals across 5 datasets + 3 modalities; one call handles all. Determinism from temp 0. |
| **Evidence retrieval** | Heuristic scoring (sender +2, group/business +1.5, Jaccard +1.5, recency bonus) | Vector embeddings | Free tier has no embeddings API; heuristic is deterministic, cheap, and surfaces exactly the right history (e.g., dismissed chain letters → mute). |
| **Context scope** | Users + groups + membership + business + history + reactions | `daily_notification_summary.csv`, raw event tables | Prompt size vs cost; summary stats add little to a single-message decision. |
| **Multimodal** | Raw image/audio bytes to one Gemini model | Separate STT/OCR pipeline (Whisper etc.) | One model for all modalities; also why DeepSeek/Groq were rejected (no native audio). Cost: media calls burn quota fast. |
| **Speed vs simplicity** | 110 sequential calls @ 4.5s apart | Batching ~10 texts/call | Batching risked per-batch JSON invalidation; sequential is resumable and predictable (~15-25 min). |
| **Resilience** | Incremental writes + resume + fallback sentinel + `--reset` | Nothing | Crash loses ≤1 message; fallback rows retried next run; `--reset` for logic changes. |
| **Prompt injection** | Prompt-level refusal of "mark notify" overrides | Dedicated injection detector | Samples confirm the model ignores overrides; zero extra cost. |
| **Evidence honesty** | Cite only real retrieved IDs, else `none` | High evidence recall | Hallucinated IDs would score worse than honest `none`; ended with 101/110 citing real history. |

## Final Result

- **110/110 messages routed** (`mute` 48 · `notify` 35 · `digest` 27)
- **101/110** decisions cite real historical evidence; 9 use `none` (no relevant history)
- All output contract checks pass: exact columns/order, valid `action` and
  `message_type`, confidence in [0,1], no empty reasons
- Code packaged as a clean `code.zip` with no venv/dataset/secrets
