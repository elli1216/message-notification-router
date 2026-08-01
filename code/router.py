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
    Extracts relevant context for a given message row using O(1) index lookups.
    """
    context = {}
    user_id = msg_row.get('user_id')
    
    # 1. User preferences
    if pd.notna(user_id):
        users_df = datasets.get('users')
        if users_df is not None and user_id in users_df.index:
            context['user_settings'] = users_df.loc[user_id].to_dict()
                
    # 2. Group Info
    group_id = msg_row.get('group_id')
    if pd.notna(group_id):
        groups_df = datasets.get('groups')
        if groups_df is not None and group_id in groups_df.index:
            context['group_info'] = groups_df.loc[group_id].to_dict()
                
        # User's group membership
        group_members_df = datasets.get('group_members')
        if group_members_df is not None and (group_id, user_id) in group_members_df.index:
            context['user_group_membership'] = group_members_df.loc[(group_id, user_id)].to_dict()

    # 3. Business Info
    business_id = msg_row.get('business_id')
    if pd.notna(business_id):
        biz_df = datasets.get('business_accounts')
        if biz_df is not None and business_id in biz_df.index:
            context['business_account_info'] = biz_df.loc[business_id].to_dict()
                
        biz_history_df = datasets.get('user_business_history')
        if biz_history_df is not None and (business_id, user_id) in biz_history_df.index:
            context['user_business_history'] = biz_history_df.loc[(business_id, user_id)].to_dict()

    return context

def _text_sim(a, b):
    ta = set(str(a).lower().split()) if pd.notna(a) else set()
    tb = set(str(b).lower().split()) if pd.notna(b) else set()
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def get_evidence(msg_row, datasets, top_k=3):
    """
    Retrieves the most relevant historical messages (and the user's reaction to
    them) for an incoming message, using O(1) index lookups where possible.
    """
    hist = datasets.get('message_history')
    if hist is None or hist.empty:
        return []
    user_id = msg_row.get('user_id')
    if pd.isna(user_id):
        return []

    incoming_ts = str(msg_row.get('created_at'))
    same_user = hist[hist['user_id'] == user_id]
    candidates = same_user[same_user['created_at'] < incoming_ts]

    sender = msg_row.get('sender_user_id')
    group_id = msg_row.get('group_id')
    biz_id = msg_row.get('business_id')
    msg_text = msg_row.get('message_text')

    scored = []
    for mid, r in candidates.iterrows():
        score = 0.0
        if pd.notna(sender) and r.get('sender_user_id') == sender:
            score += 2.0
        if pd.notna(group_id) and r.get('group_id') == group_id:
            score += 1.5
        if pd.notna(biz_id) and r.get('business_id') == biz_id:
            score += 1.5
        score += _text_sim(msg_text, r.get('message_text')) * 1.5
        try:
            days = (pd.to_datetime(incoming_ts) - pd.to_datetime(r.get('created_at'))).days
            score += max(0.0, 1.0 - days / 30.0)
        except Exception:
            pass
        scored.append((score, mid, r))

    scored.sort(key=lambda x: (-x[0], x[1]))
    events = datasets.get('message_events')

    evidence = []
    for score, mid, r in scored[:top_k]:
        item = {
            'message_id': mid,
            'created_at': r.get('created_at'),
            'message_text': str(r.get('message_text'))[:200] if pd.notna(r.get('message_text')) else '',
        }
        if pd.notna(r.get('sender_user_id')):
            item['sender_user_id'] = r.get('sender_user_id')
        if pd.notna(r.get('group_id')):
            item['group_id'] = r.get('group_id')
        if pd.notna(r.get('business_id')):
            item['business_id'] = r.get('business_id')
        if events is not None and (user_id, mid) in events.index:
            ev = events.loc[(user_id, mid)].to_dict()
            item['user_reaction'] = {k: ev[k] for k in ev if pd.notna(ev[k])}
        evidence.append(item)
    return evidence

def build_prompt(msg_row, context, evidence=None):
    """
    Constructs the prompt for the LLM.
    """
    msg_dict = msg_row.to_dict()

    evidence_section = ""
    if evidence:
        evidence_section = f"""
### HISTORICAL EVIDENCE (past messages from this user's history, with the user's reaction to each)
{json.dumps(evidence, indent=2, default=str)}
"""
    
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
For `evidence_message_ids`, list the HISTORICAL EVIDENCE message IDs that influenced your decision, separated by semicolons. Use ONLY IDs from the HISTORICAL EVIDENCE section; never invent or guess message IDs. If no historical message is relevant, output "none".
For `reason`, provide a concise, 1-2 sentence human-readable explanation of why you made this routing decision based on the context.

---

INCOMING MESSAGE:
{json.dumps(msg_dict, indent=2, default=str)}

AVAILABLE CONTEXT:
{json.dumps(context, indent=2, default=str)}
{evidence_section}"""
    return prompt

def route_message(msg_row, datasets):
    context = get_message_context(msg_row, datasets)
    evidence = get_evidence(msg_row, datasets)
    prompt = build_prompt(msg_row, context, evidence)
    
    contents = [prompt]
    
    # Multimodal Handling
    media_type = msg_row.get('media_type')
    media_id = msg_row.get('media_id')
    
    if pd.notna(media_type) and str(media_type).strip() != '' and pd.notna(media_id):
        file_path = None
        if media_type == 'image':
            img_df = datasets.get('images')
            if img_df is not None and media_id in img_df.index:
                file_path = f"../dataset/{img_df.loc[media_id]['file_path']}"
        elif media_type == 'voice':
            vn_df = datasets.get('voice_notes')
            if vn_df is not None and media_id in vn_df.index:
                file_path = f"../dataset/{vn_df.loc[media_id]['file_path']}"
                    
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
        model='gemini-3.5-flash-lite',
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
