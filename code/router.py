import os
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from data_loader import load_data

load_dotenv()
client = genai.Client()

# Define the exact output schema required by the challenge
class RoutingDecision(BaseModel):
    action: Literal["notify", "digest", "mute"] = Field(description="The final routing decision.")
    message_type: Literal["personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"] = Field(description="The best-fit message category.")
    reason: str = Field(description="A short human-readable explanation for the decision.")
    confidence: float = Field(description="Confidence score from 0 to 1.")
    evidence_message_ids: str = Field(description="Semicolon-separated historical message IDs used as evidence, or 'none'.")

def get_message_context(msg_row, datasets):
    """
    Extracts relevant context for a given message row.
    """
    context = {}
    user_id = msg_row.get('user_id')
    
    # 1. User preferences
    if pd.notna(user_id):
        users_df = datasets.get('users')
        if users_df is not None:
            user_info = users_df[users_df['user_id'] == user_id]
            if not user_info.empty:
                context['user_settings'] = user_info.iloc[0].to_dict()
                
    # 2. Group Info
    group_id = msg_row.get('group_id')
    if pd.notna(group_id):
        groups_df = datasets.get('groups')
        if groups_df is not None:
            group_info = groups_df[groups_df['group_id'] == group_id]
            if not group_info.empty:
                context['group_info'] = group_info.iloc[0].to_dict()
                
        # User's group membership
        group_members_df = datasets.get('group_members')
        if group_members_df is not None and pd.notna(user_id):
            membership = group_members_df[(group_members_df['group_id'] == group_id) & (group_members_df['user_id'] == user_id)]
            if not membership.empty:
                context['user_group_membership'] = membership.iloc[0].to_dict()

    # 3. Business Info
    business_id = msg_row.get('business_id')
    if pd.notna(business_id):
        biz_df = datasets.get('business_accounts')
        if biz_df is not None:
            biz_info = biz_df[biz_df['business_id'] == business_id]
            if not biz_info.empty:
                context['business_account_info'] = biz_info.iloc[0].to_dict()
                
        biz_history_df = datasets.get('user_business_history')
        if biz_history_df is not None and pd.notna(user_id):
            history = biz_history_df[(biz_history_df['business_id'] == business_id) & (biz_history_df['user_id'] == user_id)]
            if not history.empty:
                context['user_business_history'] = history.iloc[0].to_dict()

    return context

def build_prompt(msg_row, context):
    """
    Constructs the prompt for the LLM.
    """
    msg_dict = msg_row.to_dict()
    
    prompt = f"""You are an expert AI system designed to route incoming WhatsApp messages. Your goal is to protect the user's attention by deciding if a message requires immediate interruption (`notify`), can be batched for later (`digest`), or should be suppressed entirely (`mute`).

You will receive an INCOMING MESSAGE and its associated AVAILABLE CONTEXT (which may include user preferences, group metadata, and business history). If media (images or voice notes) are attached, analyze their contents carefully.

### DECISION LOGIC & RULES

1. **`notify` (Interrupt Now)**
   - Urgent personal messages, emergencies, or direct mentions.
   - Time-sensitive updates (e.g., OTPs, delivery arrivals, immediate meeting changes).
   - Important updates from trusted groups (e.g., school closures, admin announcements).

2. **`digest` (Read Later)**
   - Low-priority group chatter, casual greetings, or non-urgent personal updates.
   - General business updates, newsletters, or subscribed promotions (if the user allows promotions).
   - Event reminders that are not immediate.

3. **`mute` (Suppress/Block)**
   - Clear spam, scams, phishing attempts, or unsafe content.
   - Promotions from businesses the user has opted out of or reported.
   - Messages from groups the user has explicitly muted (unless they are directly @mentioned or it is a critical admin broadcast).
   - Highly forwarded generic messages (e.g., "Good morning" images) unless specifically valued by the user in their history.

### OUTPUT REQUIREMENTS
You must output valid JSON matching the provided schema. Carefully determine the best `message_type` from the allowed list based on the text and/or media content. 
For `evidence_message_ids`, list any historical message IDs that influenced your decision, separated by semicolons. If none apply, output "none".
For `reason`, provide a concise, 1-2 sentence human-readable explanation of why you made this routing decision based on the context.

---

INCOMING MESSAGE:
{json.dumps(msg_dict, indent=2, default=str)}

AVAILABLE CONTEXT:
{json.dumps(context, indent=2, default=str)}
"""
    return prompt

def route_message(msg_row, datasets):
    context = get_message_context(msg_row, datasets)
    prompt = build_prompt(msg_row, context)
    
    contents = [prompt]
    
    # Multimodal Handling
    media_type = msg_row.get('media_type')
    media_id = msg_row.get('media_id')
    
    if pd.notna(media_type) and str(media_type).strip() != '' and pd.notna(media_id):
        file_path = None
        if media_type == 'image':
            img_df = datasets.get('images')
            if img_df is not None:
                row = img_df[img_df['image_id'] == media_id]
                if not row.empty:
                    file_path = f"../dataset/{row.iloc[0]['file_path']}"
        elif media_type == 'voice':
            vn_df = datasets.get('voice_notes')
            if vn_df is not None:
                row = vn_df[vn_df['voice_note_id'] == media_id]
                if not row.empty:
                    file_path = f"../dataset/{row.iloc[0]['file_path']}"
                    
        if file_path and os.path.exists(file_path):
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'image/jpeg' if media_type == 'image' else 'audio/mp3'
            
            print(f"  [Attaching media: {file_path}]")
            with open(file_path, "rb") as f:
                data = f.read()
            contents.append(
                types.Part.from_bytes(data=data, mime_type=mime_type)
            )
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RoutingDecision,
            temperature=0.0
        ),
    )
    
    # Validate and parse the returned JSON string into our Pydantic model
    return RoutingDecision.model_validate_json(response.text)

if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_data('../dataset')
    
    sample_df = datasets.get('sample_messages')
    if sample_df is not None and not sample_df.empty:
        print("\nTesting router on select sample messages (Text, Image, Voice)...")
        
        # Find one of each type for testing
        text_msgs = sample_df[sample_df['media_type'].isna() | (sample_df['media_type'] == '')]
        img_msgs = sample_df[sample_df['media_type'] == 'image']
        voice_msgs = sample_df[sample_df['media_type'] == 'voice']
        
        test_cases = []
        if not text_msgs.empty: test_cases.append(text_msgs.iloc[0])
        if not img_msgs.empty: test_cases.append(img_msgs.iloc[0])
        if not voice_msgs.empty: test_cases.append(voice_msgs.iloc[0])
        
        for i, test_msg in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}: Media Type = {test_msg['media_type']} ---")
            try:
                decision = route_message(test_msg, datasets)
                print(f"Action:       {decision.action}")
                print(f"Type:         {decision.message_type}")
                print(f"Reason:       {decision.reason}")
                
                print(f"Ground Truth: {test_msg['action']} | {test_msg['message_type']}")
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
