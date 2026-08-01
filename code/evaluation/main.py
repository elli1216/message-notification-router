import pandas as pd
from data_loader import load_data
from router import route_message
import time

def evaluate_on_samples():
    print("Loading datasets for evaluation...")
    datasets = load_data('../dataset')
    
    sample_df = datasets.get('sample_messages')
    if sample_df is None or sample_df.empty:
        print("No sample messages found.")
        return
        
    print(f"Starting evaluation on {len(sample_df)} ground-truth sample messages...")
    
    correct_actions = 0
    correct_types = 0
    total = len(sample_df)
    
    for index, msg_row in sample_df.iterrows():
        msg_id = msg_row['message_id']
        ground_truth_action = msg_row['action']
        ground_truth_type = msg_row['message_type']
        
        try:
            decision = route_message(msg_row, datasets)
            
            action_match = decision.action == ground_truth_action
            type_match = decision.message_type == ground_truth_type
            
            if action_match: correct_actions += 1
            if type_match: correct_types += 1
            
            print(f"[{'PASS' if action_match else 'FAIL'}] {msg_id}: "
                  f"Predicted={decision.action} ({decision.message_type}) | "
                  f"Actual={ground_truth_action} ({ground_truth_type})")
                  
        except Exception as e:
            print(f"[ERROR] failed on {msg_id}: {e}")
            
        time.sleep(0.5) # respect rate limits during eval
            
    print("\n--- Evaluation Results ---")
    print(f"Action Accuracy:       {correct_actions}/{total} ({(correct_actions/total)*100:.2f}%)")
    print(f"Message Type Accuracy: {correct_types}/{total} ({(correct_types/total)*100:.2f}%)")

if __name__ == "__main__":
    evaluate_on_samples()
