import os
import requests
import json
import time
import random

# Ekkada mi API settings config cheyochu
# 'dummy' vadithe LLM API lekunda random ga select chesthundi (loop test cheyadaniki)
# 'openai' vadithe real gpt-3.5 or gpt-4 model connect avthundi
LLM_PROVIDER = "dummy" 
API_KEY = "YOUR_API_KEY_HERE"

BASE_URL = "http://localhost:8000/api"

def get_current_state():
    """Step 1: Backend API nunchi current State data theeskovadam"""
    response = requests.get(f"{BASE_URL}/state")
    return response.json()

def execute_action(action_idx):
    """Step 4: LLM select chesina action ni backend ki pampadam"""
    response = requests.post(f"{BASE_URL}/step", json={"action_idx": action_idx})
    return response.json()

def ask_llm_for_action(state_data):
    """Step 2: State data ni prompt la marchi, LLM ni asugu (Step 3)"""
    state_metrics = state_data["state"]
    revenue = state_metrics["revenue"]
    cash = state_metrics["cash_balance"]
    morale = state_metrics["employee_morale"]
    
    # Manam Create chesthunna English Prompt
    prompt = f"""You are the CPU/CEO of ATLAS.
Company Metrics:
- Revenue: ${revenue}
- Cash: ${cash}
- Employee Morale: {morale}/10

Your available actions:
0 - Relax
1 - Start New Feature
2 - Review Code
3 - Outbound Campaign
4 - Customer Followups
5 - Post Job Ad
6 - Conduct Interviews
7 - Team Building Event
8 - Review Financials
9 - Cut Costs
10 - Customer Support

Reply with ONLY a single number from 0 to 10."""

    print("\n--- Sending Prompt to LLM ---")
    print(prompt)
    print("-----------------------------\n")

    if LLM_PROVIDER == "dummy":
        # Meeru real API key pette varaku idi random ga numbers theeskuntundi
        return random.randint(0, 10)
        
    elif LLM_PROVIDER == "huggingface":
        # Hugging Face Free Inference API
        # Mistral-7B-Instruct model ni direct ga vaadatam
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Mistral format lo prompt istham
        hf_prompt = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "inputs": hf_prompt,
            "parameters": {"max_new_tokens": 5, "temperature": 0.1, "return_full_text": False}
        }
        
        try:
            hf_response = requests.post(API_URL, headers=headers, json=payload)
            hf_response.raise_for_status()
            prediction = hf_response.json()[0]["generated_text"].strip()
            # Numbers matrame filter cheyadam
            digits = "".join([c for c in prediction if c.isdigit()])
            if digits:
                return int(digits)
            return 0
        except Exception as e:
            print(f"HuggingFace API Error: {e}")
            return random.randint(0, 10)
    
    elif LLM_PROVIDER == "openai":
        # Meeru install cheyali: pip install openai
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.5
        )
        prediction = response.choices[0].message.content.strip()
        # Answer nunchi letters theeseyadam, just number ni retain cheyyadam
        digits = "".join([c for c in prediction if c.isdigit()])
        if digits:
            return int(digits)
        return 0

    return 0

def run_llm_simulation():
    print("Starting LLM CEO Simulation...")
    # First environment ni start or reset chestham
    requests.post(f"{BASE_URL}/reset", json={"preset": "startup"})
    
    while True:
        # Loop starts for 90 days!
        state_data = get_current_state()
        
        # 90 Days aipoyaaya check cheyadam
        if state_data.get("done", False):
            print(f"\nSimulation Finished! Final Score/Points: {state_data.get('reward', 0)}")
            break
            
        # LLM daggara decision theeskovadam
        action_idx = ask_llm_for_action(state_data)
        print(f"-> LLM Chose Action Index: {action_idx}")
        
        # Action ni Project lo Run cheyadam
        execute_action(action_idx)
        
        # 3 seconds aagadam (so that meeru live dashboard chudagalaru)
        time.sleep(3) 

if __name__ == "__main__":
    run_llm_simulation()
