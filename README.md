# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

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
    ├── output.csv                    # Submission template / predictions
    ├── sample_messages.csv           # Solved examples
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
approximately 15-25 minutes. A partial `output.csv` can be submitted while the
rest is processed, as long as all rows are filled by the final run.

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

## What You Need to Build

For every row in `dataset/messages.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | Best-fit message category |
| `reason` | Short human-readable explanation |
| `confidence` | Number from `0` to `1` |
| `evidence_message_ids` | Historical message IDs used as evidence; write `none` if there is no useful evidence |

Your system should make personalized decisions using the provided message, user, group, business, media, and historical interaction data.
For image and voice-note messages, `images.csv` and `voice_notes.csv` only provide file paths; your system should inspect the media files themselves.

---

## Suggested Workflow

1. Inspect `dataset/sample_messages.csv` to understand the expected output format.
2. Load `dataset/messages.csv` and all relevant context files.
3. Build your routing system using any approach: LLMs, retrieval, rules, classifiers, agents, or hybrids.
4. Write predictions to `output.csv`.
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

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
