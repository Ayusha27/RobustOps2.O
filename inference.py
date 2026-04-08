import os
import asyncio
import ast
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from openai import OpenAI
from env.environment import RobustOpsEnv
from env.models import Action

# Load local environment variables
load_dotenv()

# --- 1. Configuration & Validation ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = "phishing-detection"
BENCHMARK = "robust-ops"

if not HF_TOKEN or HF_TOKEN.strip() == "":
    raise ValueError("HF_TOKEN is required but not found in environment variables.")

# Initialize OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# --- 2. FastAPI Setup for Validator Compliance ---
app = FastAPI()

@app.get("/")
async def health():
    """Basic health check for the Space."""
    return {"status": "alive"}

@app.post("/reset")
async def reset_endpoint():
    """
    Satisfies 'Step 1/3: Pinging HF Space (/reset)'.
    Triggers the inference logic in the background.
    """
    asyncio.create_task(run_submission())
    return {"status": "success", "message": "Environment reset"}

# --- 3. Core Logic & Logging ---
async def get_llm_decision(signals):
    """Added deeper error handling and retries."""
    prompt = (
        f"Analyze these phishing signals: {signals}. "
        "Respond with exactly one word: 'spam', 'not_spam', or 'uncertain'."
    )
    for attempt in range(3): # Try 3 times before giving up
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                timeout=15.0 # Don't let the LLM hang forever
            )
            if not response.choices:
                continue
                
            answer = response.choices[0].message.content.lower()
            if "uncertain" in answer: return None
            return "spam" if "spam" in answer else "not_spam"
        except Exception as e:
            print(f"LLM Attempt {attempt+1} failed: {e}", flush=True)
            await asyncio.sleep(1) # Wait a second before retrying
    return None

async def run_submission():
    """Ensures a clean exit and logs every error."""
    steps_taken = 0
    score = 0.0
    success = False
    rewards = []
    
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    try:
        # Wrap the env init in a try-block
        try:
            env = RobustOpsEnv()
            obs = env.reset()
        except Exception as e:
            print(f"Environment Reset Failed: {e}", flush=True)
            return # Exit if we can't even start

        for step in range(1, 3):
            try:
                # 1. Parse signals carefully
                signals = ["urgent_tone"]
                if "Signals: " in obs.message:
                    signals_str = obs.message.split("Signals: ")[-1]
                    signals = ast.literal_eval(signals_str)

                # 2. Get decision
                decision = await get_llm_decision(signals) # Make sure this is awaited if using async
                
                action_type = "flag_uncertain" if decision is None else ("classify" if step == 1 else "revise")
                action_content = decision

                # 3. Step interaction
                obs, reward_obj, done, info = env.step(Action(action_type=action_type, content=action_content))
                
                reward = reward_obj.value
                rewards.append(reward)
                steps_taken = step
                
                print(f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

                if done: break
            except Exception as step_error:
                print(f"[STEP] step={step} action=error reward=0.00 done=true error={type(step_error).__name__}", flush=True)
                break

        score = info.get("score", 0.0) if 'info' in locals() else 0.0
        success = score >= 0.5

    except Exception as fatal_error:
        print(f"Fatal Traceback: {fatal_error}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={rewards_str}", flush=True)
def main_server():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main_server()