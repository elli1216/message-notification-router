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
    
    prompt = f"""You are a highly intelligent WhatsApp Message Notification Router.
Your task is to decide whether the incoming message should trigger a 'notify' (interrupt user), 'digest' (show later), or 'mute' (suppress).

INCOMING MESSAGE:
{json.dumps(msg_dict, indent=2, default=str)}

AVAILABLE CONTEXT (User Settings, Group Info, Business History):
{json.dumps(context, indent=2, default=str)}

Based on the INCOMING MESSAGE and AVAILABLE CONTEXT, make a routing decision. 
Take into account the user's settings, any group settings, and their history with this business/sender.
If it's a scam or clearly dangerous, mute it.
"""
    return prompt

def route_message(msg_row, datasets):
    context = get_message_context(msg_row, datasets)
    prompt = build_prompt(msg_row, context)
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
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
        print("\nTesting router on the first sample message with Gemini-3.5-Flash...")
        test_msg = sample_df.iloc[0]
        
        try:
            decision = route_message(test_msg, datasets)
            print("\n=== LLM Routing Decision ===")
            print(f"Action:       {decision.action}")
            print(f"Type:         {decision.message_type}")
            print(f"Reason:       {decision.reason}")
            print(f"Confidence:   {decision.confidence}")
            print(f"Evidence IDs: {decision.evidence_message_ids}")
            
            print("\n=== Actual Ground Truth ===")
            print(f"Action:       {test_msg['action']}")
            print(f"Type:         {test_msg['message_type']}")
            print(f"Reason:       {test_msg['reason']}")
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
