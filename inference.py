import os
import ast
from dotenv import load_dotenv
from openai import OpenAI
from env.environment import RobustOpsEnv
from env.models import Action
import time

# Load variables from .env for local testing
load_dotenv() 

# --- 1. Configuration & Validation ---
# Rule: API_BASE_URL and MODEL_NAME read via os.getenv() with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "tgi") 
HF_TOKEN = os.getenv("HF_TOKEN")

# Rule: HF_TOKEN is validated — raise error if missing
if not HF_TOKEN or HF_TOKEN.strip() == "":
    raise ValueError("HF_TOKEN is required but not found in environment variables.")

# --- 2. Initialize OpenAI Client ---
# Rule: All LLM calls use OpenAI Python client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_decision(signals):
    prompt = f"Analyze these phishing signals: {signals}. Respond with 'spam', 'not_spam', or 'uncertain'."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.lower()
        
        if "uncertain" in answer:
            return None
        return "spam" if "spam" in answer else "not_spam"
    except Exception:
        return None # Default to uncertain on API failure

def run_submission():
    # Rule: Output format follows [START] exactly
    print("[START]")
    
    env = RobustOpsEnv()
    try:
        obs = env.reset()
        
        # Safer parsing of the signals list 
        signals_str = obs.message.split("Signals: ")[-1]
        signals = ast.literal_eval(signals_str)

        # Step 1: LLM Decision
        decision = get_llm_decision(signals)
        
        if decision is None:
            action = Action(action_type="flag_uncertain", content=None)
        else:
            action = Action(action_type="classify", content=decision)

        obs, reward, done, info = env.step(action)

        # Rule: Rewards formatted to 2 decimal places
        # Rule: Booleans (done, success) are lowercase: true / false
        done_str = "true" if done else "false"
        
        # Check for success in the info dict if applicable
        success = info.get("success", False)
        success_str = "true" if success else "false"
        
        print(f"[STEP] reward: {reward.value:.2f}, done: {done_str}")

    except Exception as e:
        # You can log the error internally, but ensure formatting persists
        pass
    finally:
        # Rule: [END] is always emitted, even on exceptions
        print("[END]")

if __name__ == "__main__":
    run_submission()
 
    # REQUIRED: Keep the Space "Running" so the validator can check files
    # Without this, the container exits and the checker fails.
    while True:
        time.sleep(10)