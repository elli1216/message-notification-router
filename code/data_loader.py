import pandas as pd
import os
from pathlib import Path

def load_data(base_path='../dataset'):
    """Loads all dataset CSV files into a dictionary of DataFrames."""
    base = Path(base_path)
    data = {}
    
    files = [
        'messages.csv', 'sample_messages.csv', 'users.csv', 'groups.csv',
        'group_members.csv', 'business_accounts.csv', 'user_business_history.csv',
        'message_history.csv', 'message_events.csv', 'images.csv', 'voice_notes.csv',
        'daily_notification_summary.csv'
    ]
    
    for f in files:
        file_path = base / f
        if file_path.exists():
            key = f.replace('.csv', '')
            data[key] = pd.read_csv(file_path)
        else:
            print(f"Warning: {f} not found at {file_path}")
            
    return data

if __name__ == "__main__":
    print("Testing data loader...")
    # Assumes this script is run from the 'code' directory
    datasets = load_data('../dataset')
    
    if 'messages' in datasets:
        print(f"Successfully loaded {len(datasets['messages'])} messages to route.")
    
    if 'sample_messages' in datasets:
        print(f"Successfully loaded {len(datasets['sample_messages'])} sample (ground truth) messages.")
