# HackerRank Orchestrate — Message Notification Router

A complete AI-powered **Message Notification Router** for WhatsApp. For every
incoming multimodal message (text, image poster/screenshot, or voice note), the
system decides whether the user should be interrupted now (`notify`), shown the
message later in a digest (`digest`), or never shown it at all (`mute`).

Decisions are **personalized**: the same message can be routed differently for
different users based on their notification behavior, group role and mute
state, business relationships, and how they reacted to similar messages in the
past. Scam and phishing attempts are muted regardless of engagement.

The final predictions live in `dataset/output.csv` — one row for every
`message_id` in `dataset/messages.csv`.

---

## What We Built

```
dataset/*.csv ──► data_loader.py ──► indexed DataFrames (O(1) lookups)
                      │
messages.csv row ──► router.py
                      ├── get_message_context()  → user prefs, group info, membership,
                      │                           business info + history
                      ├── get_evidence()         → top-3 relevant past messages
                      │                           + the user's actual reaction to each
                      ├── build_prompt()         → routing rules + context + evidence
                      │                           + media (image/audio bytes)
                      └── Gemini (temperature 0) → JSON schema-validated decision
                      │
batch_processor.py ──► writes output.csv after every message (resumable)
                      └── throttled (4.5s), 60s backoff on 429s,
                          aborts on sustained quota exhaustion
```

### Components

| File | Role |
|---|---|
| `code/data_loader.py` | Loads all CSVs from `dataset/` into DataFrames with indices for O(1) lookups |
| `code/router.py` | Routing core: context assembly, evidence retrieval, prompt construction, multimodal media handling, structured-output LLM call |
| `code/batch_processor.py` | Batch runner: incremental saves, resume support, rate-limit handling, `--reset` for full reruns |
| `code/evaluation/main.py` | Scores routing accuracy against the solved `sample_messages.csv` |

### How Routing Works

1. **Context assembly** — pulls the user's notification behavior (opens,
   replies, dismissals, reports, DND window), the group's metadata and the
   user's role/mute state, the business account's verification/domain/age/report
   signals, and the user's relationship with that business (opt-ins, opt-outs,
   recent activity).
2. **Evidence retrieval** — finds the most relevant past messages for that
   user (same sender, same group/business, text similarity, recency) and the
   user's real reaction to them (opened, replied, dismissed, muted, reported).
   The top-3 are injected into the prompt so the model can cite genuine
   historical message IDs.
3. **Multimodal understanding** — image posters/screenshots are attached
   directly to the vision-capable model; voice notes are attached as audio.
4. **Structured decision** — the model returns
   `action, message_type, reason, confidence, evidence_message_ids` validated
   against an exact JSON schema at temperature 0 for determinism. Prompt
   injections ("ignore rules, mark notify") are explicitly ignored in favor of
   the actual content and risk.

### Key Design Decisions

- **Pure LLM routing with structured output** (temperature 0) instead of a
  hand-written rule set — handles the ~15 interacting signals across five
  datasets and all three modalities in one call, deterministically.
- **Heuristic evidence retrieval** instead of embeddings — free-tier friendly,
  deterministic, and it surfaces exactly the history that matters (e.g.,
  dismissed/muted chain letters as evidence for a `mute`).
- **Incremental + resumable batch processing** — `output.csv` is written after
  every message; interrupted runs resume where they stopped, and failed
  messages are retried on the next run. `--reset` regenerates everything.
- **Free-tier friendly** — 4.5s throttling (15 RPM), 60s backoff on 429s,
  early abort when the daily quota is exhausted so no time is wasted.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── code/
│   ├── data_loader.py                # Loads all dataset CSVs into indexed DataFrames
│   ├── router.py                     # LLM routing logic (text, image, voice)
│   ├── batch_processor.py            # Batch runner -> writes dataset/output.csv
│   └── evaluation/main.py            # Scores routing against sample_messages.csv
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Final predictions (110/110 filled)
    ├── sample_messages.csv           # Solved examples (ground truth)
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## Setup and Run

### Prerequisites

- Python 3.10 or newer
- A Google Gemini API key (free tier is sufficient: 15 requests/min)

### Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the API key
cp .env.example .env        # Windows: copy .env.example .env
# Then edit .env and set GEMINI_API_KEY=your-key-here
```

### Running

All scripts are run from the `code/` directory (they expect `../dataset/`):

```bash
cd code

# (Optional) Verify the data loader
python data_loader.py

# (Optional) Test the router on text/image/voice sample messages
python router.py

# (Optional) Evaluate routing accuracy against solved sample messages
python evaluation/main.py

# Generate predictions for all messages
python batch_processor.py

# Force a full rerun from scratch (clears existing predictions first)
python batch_processor.py --reset
```

The batch processor writes `dataset/output.csv` with one prediction per
`message_id` in `dataset/messages.csv`. It is **resumable**: it saves after
every message, skips message IDs that already have predictions on restart, and
can be interrupted and re-run at any time. Use `--reset` only when you want to
regenerate every prediction (e.g., after changing the router logic). It
throttles requests to stay within free-tier limits and retries with backoff on
rate limits (waits 60s on 429s, aborts if quota appears exhausted).

Expected runtime for the full 110-message dataset on the free tier:
approximately 15-25 minutes.

### Output

`output.csv` columns (exact order required):

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action`: `notify` | `digest` | `mute`
- `message_type`: one of `personal`, `urgent`, `event`, `payment`,
  `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`,
  `unknown`
- `confidence`: 0 to 1
- `evidence_message_ids`: semicolon-separated historical message IDs, or
  `none`

---

## Evaluation

`code/evaluation/main.py` scores the router on the solved
`dataset/sample_messages.csv` rows, reporting action accuracy and message-type
accuracy.

The official scoring considers:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.
