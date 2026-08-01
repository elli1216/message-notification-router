import pandas as pd
import time
from router import route_message
from data_loader import load_data

def process_all_messages():
    print("Loading datasets...")
    datasets = load_data('../dataset')
    
    messages_df = datasets.get('messages')
    if messages_df is None or messages_df.empty:
        print("No messages found in dataset/messages.csv")
        return
        
    print(f"Starting batch processing of {len(messages_df)} messages...")
    
    results = []
    
    for index, msg_row in messages_df.iterrows():
        msg_id = msg_row['message_id']
        print(f"Processing {msg_id} ({index + 1}/{len(messages_df)})...")
        
        retries = 3
        wait_time = 2
        while retries > 0:
            try:
                decision = route_message(msg_row, datasets)
                results.append({
                    'message_id': msg_id,
                    'action': decision.action,
                    'message_type': decision.message_type,
                    'reason': decision.reason,
                    'confidence': decision.confidence,
                    'evidence_message_ids': decision.evidence_message_ids
                })
                break
            except Exception as e:
                retries -= 1
                print(f"  [!] Error processing {msg_id}: {e}. Retries left: {retries}")
                if retries > 0:
                    time.sleep(wait_time)
                    wait_time *= 2 # Exponential backoff
                else:
                    # Safe fallback so we don't break the entire output file
                    print(f"  [!] Failed to process {msg_id}. Applying fallback.")
                    results.append({
                        'message_id': msg_id,
                        'action': 'digest',
                        'message_type': 'unknown',
                        'reason': 'API failure during batch processing.',
                        'confidence': 0.0,
                        'evidence_message_ids': 'none'
                    })
                    
        # Small delay to respect API rate limits
        time.sleep(0.5)

    print("\nFinished processing all messages.")
    
    # Create final DataFrame
    output_df = pd.DataFrame(results)
    
    # Enforce exact column order required by the project contract
    cols = ['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']
    output_df = output_df[cols]
    
    output_path = '../dataset/output.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Successfully saved all predictions to {output_path}")

if __name__ == "__main__":
    process_all_messages()
