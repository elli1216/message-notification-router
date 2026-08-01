import pandas as pd
import sys
import time
from pathlib import Path
from router import route_message
from data_loader import load_data

COLS = ['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']
OUTPUT_PATH = '../dataset/output.csv'
SLEEP_BETWEEN = 4.5
RATE_LIMIT_WAIT = 60
MAX_ATTEMPTS = 5

def is_rate_limit(e):
    code = getattr(e, 'code', None)
    if code == 429:
        return True
    text = str(e)
    return ('RESOURCE_EXHAUSTED' in text.upper()
            or '429' in text
            or 'quota' in text.lower()
            or 'rate limit' in text.lower())

def load_completed(output_path):
    if not Path(output_path).exists():
        return {}
    try:
        df = pd.read_csv(output_path)
    except Exception:
        return {}
    done = {}
    for _, row in df.iterrows():
        mid = str(row['message_id'])
        action = row.get('action')
        if pd.notna(action) and str(action).strip() != '' \
                and str(row.get('reason', '')).strip() != 'API failure during batch processing.':
            done[mid] = {c: row.get(c) for c in COLS}
    return done

def get_template_order(output_path, messages_df):
    if Path(output_path).exists():
        try:
            df = pd.read_csv(output_path, usecols=['message_id'])
            return [str(x) for x in df['message_id'].tolist()]
        except Exception:
            pass
    return [str(r['message_id']) for _, r in messages_df.iterrows()]

def write_output(results, template_order, just_written):
    rows = []
    for mid in template_order:
        row = results.get(mid)
        if row is None:
            rows.append({'message_id': mid})
        else:
            rows.append({c: row.get(c, '') for c in COLS})
    out = pd.DataFrame(rows)[COLS]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"  [saved] output.csv updated (last written: {just_written})")

def reset_output(template_order):
    rows = [{'message_id': mid} for mid in template_order]
    out = pd.DataFrame(rows)
    for c in COLS[1:]:
        out[c] = ''
    out = out[COLS]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[reset] output.csv cleared ({len(rows)} rows)")

def process_all_messages(reset=False):
    print("Loading datasets...")
    datasets = load_data('../dataset')

    messages_df = datasets.get('messages')
    if messages_df is None or messages_df.empty:
        print("No messages found in dataset/messages.csv")
        return

    template_order = get_template_order(OUTPUT_PATH, messages_df)
    if reset:
        reset_output(template_order)
        completed = {}
    else:
        completed = load_completed(OUTPUT_PATH)
    todo = [r for _, r in messages_df.iterrows() if str(r['message_id']) not in completed]
    print(f"Total: {len(messages_df)} | Already done: {len(completed)} | To process: {len(todo)}")

    results = dict(completed)

    consecutive_rate_limited = 0

    for i, msg_row in enumerate(todo):
        msg_id = str(msg_row['message_id'])
        print(f"Processing {msg_id} ({i + 1}/{len(todo)})...")

        decision = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                decision = route_message(msg_row, datasets)
                break
            except Exception as e:
                if is_rate_limit(e):
                    consecutive_rate_limited += 1
                    print(f"  [!] Rate limited on {msg_id}: {e}. Waiting {RATE_LIMIT_WAIT}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                    time.sleep(RATE_LIMIT_WAIT)
                    if consecutive_rate_limited >= 3:
                        print("[!] Sustained rate limiting - likely daily quota exhausted. Progress is saved; aborting.")
                        write_output(results, template_order, msg_id)
                        raise SystemExit(1)
                else:
                    wait = 5 * (attempt + 1)
                    print(f"  [!] Error on {msg_id}: {e}. Retrying in {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                    time.sleep(wait)

        if decision is not None:
            consecutive_rate_limited = 0

        if decision is None:
            print(f"  [!] Failed to process {msg_id}. Applying fallback.")
            row = {
                'message_id': msg_id,
                'action': 'digest',
                'message_type': 'unknown',
                'reason': 'API failure during batch processing.',
                'confidence': 0.0,
                'evidence_message_ids': 'none'
            }
        else:
            row = {
                'message_id': msg_id,
                'action': decision.action,
                'message_type': decision.message_type,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'evidence_message_ids': decision.evidence_message_ids
            }

        results[msg_id] = row
        write_output(results, template_order, msg_id)
        print(f"  -> {msg_id}: {row['action']} ({row['message_type']}) conf={row['confidence']}")

        time.sleep(SLEEP_BETWEEN)

    print("\nFinished processing all messages.")
    write_output(results, template_order, 'all')
    print(f"Final predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_messages(reset='--reset' in sys.argv)
