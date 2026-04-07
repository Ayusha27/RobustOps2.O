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
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct")
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
def get_llm_decision(signals):
    """Encapsulated LLM logic using the OpenAI client."""
    prompt = (
        f"Analyze these phishing signals: {signals}. "
        "Respond with exactly one word: 'spam', 'not_spam', or 'uncertain'."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        answer = response.choices[0].message.content.lower()
        if "uncertain" in answer: return None
        return "spam" if "spam" in answer else "not_spam"
    except Exception:
        return None

async def run_submission():
    """Main execution logic with mandatory stdout formatting."""
    steps_taken = 0
    score = 0.0
    success = False
    rewards = []
    
    # Rule: [START] line
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    env = RobustOpsEnv()
    
    try:
        obs = env.reset()
        for step in range(1, 3):
            # Parse signals
            try:
                signals_str = obs.message.split("Signals: ")[-1]
                signals = ast.literal_eval(signals_str)
            except:
                signals = ["urgent_tone"]

            decision = get_llm_decision(signals)
            
            if decision is None:
                action_type, action_content = "flag_uncertain", None
            else:
                action_type = "classify" if step == 1 else "revise"
                action_content = decision

            # Step interaction
            obs, reward_obj, done, info = env.step(Action(action_type=action_type, content=action_content))
            
            reward = reward_obj.value
            rewards.append(reward)
            steps_taken = step
            
            # Rule: [STEP] line with 2 decimal rewards and lowercase booleans
            print(f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            if done: break

        score = info.get("score", 0.0)
        success = score >= 0.5

    except Exception:
        pass
    finally:
        # Rule: [END] always emitted
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    # Start server on port 7860 (Hugging Face default)
    uvicorn.run(app, host="0.0.0.0", port=7860)