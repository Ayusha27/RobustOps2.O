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

# Initialize OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "alive", "model": MODEL_NAME}

@app.post("/reset")
async def reset_endpoint():
    try:
        results = await run_submission()
        return {"status": "success", "data": results}
    except Exception as e:
        print(f"Global Endpoint Error: {e}", flush=True)
        return {"status": "error", "message": str(e)}

# --- 2. Core Logic ---

async def get_llm_decision(signals):
    """Refined prompt for strict output and high scores."""
    prompt = (
        f"Signals: {signals}. Task: Is this phishing? "
        "Respond ONLY with 'spam' or 'not_spam'. No punctuation."
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert security analyst. Reply only with 'spam' or 'not_spam'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0,
                timeout=15.0
            )
            answer = response.choices[0].message.content.lower().strip()
            if "not_spam" in answer: return "not_spam"
            if "spam" in answer: return "spam"
        except Exception as e:
            print(f"LLM Error (Attempt {attempt+1}): {e}", flush=True)
            await asyncio.sleep(1)
    return None

async def run_submission():
    """Main execution logic with mandatory stdout formatting."""
    steps_taken = 0
    score = 0.0
    success = False
    rewards = []
    
    # Rule: Always emit [START]
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    try:
        if not HF_TOKEN:
            print("Error: HF_TOKEN missing", flush=True)
            return

        env = RobustOpsEnv()
        obs = env.reset()
        
        for step in range(1, 3):
            # Parse signals
            try:
                signals = ["urgent_tone"]
                if "Signals: " in obs.message:
                    signals_str = obs.message.split("Signals: ")[-1]
                    signals = ast.literal_eval(signals_str)
            except:
                signals = ["urgent_tone"]

            decision = await get_llm_decision(signals)
            
            # Action logic
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
            
            # Rule: [STEP] line formatting
            print(f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            if done: break

        score = info.get("score", 0.0)
        success = score >= 0.5

    except Exception as e:
        print(f"Runtime Error: {e}", flush=True)
    finally:
        # Rule: [END] must always be emitted
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={rewards_str}", flush=True)

    return {"score": score, "success": success}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)