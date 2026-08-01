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
            df = pd.read_csv(file_path)
            
            # Set indices for O(1) lookups
            if key == 'users' and 'user_id' in df.columns:
                df.set_index('user_id', inplace=True)
            elif key == 'groups' and 'group_id' in df.columns:
                df.set_index('group_id', inplace=True)
            elif key == 'business_accounts' and 'business_id' in df.columns:
                df.set_index('business_id', inplace=True)
            elif key == 'group_members' and 'group_id' in df.columns and 'user_id' in df.columns:
                df.set_index(['group_id', 'user_id'], inplace=True)
            elif key == 'user_business_history' and 'business_id' in df.columns and 'user_id' in df.columns:
                df.set_index(['business_id', 'user_id'], inplace=True)
            elif key == 'images' and 'image_id' in df.columns:
                df.set_index('image_id', inplace=True)
            elif key == 'voice_notes' and 'voice_note_id' in df.columns:
                df.set_index('voice_note_id', inplace=True)
            elif key == 'message_history' and 'message_id' in df.columns:
                df.set_index('message_id', inplace=True)
            elif key == 'message_events' and 'user_id' in df.columns and 'message_id' in df.columns:
                df.set_index(['user_id', 'message_id'], inplace=True)
                
            data[key] = df
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
