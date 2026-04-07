from env.environment import RobustOpsEnv
from env.models import Action

env = RobustOpsEnv()

# Reset
obs = env.reset()
print("Initial:", obs)

# Step 1
action1 = Action(action_type="classify", content="important")
obs, reward, done, info = env.step(action1)

print("\n--- Step 1 ---")
print("Observation:", obs)
print("Reward:", reward)
print("Done:", done)
print("Score:", info.get("score"))

# Step 2
action2 = Action(action_type="revise", content="spam")
obs, reward, done, info = env.step(action2)

print("\n--- Step 2 ---")
print("Observation:", obs)
print("Reward:", reward)
print("Done:", done)
print("Score:", info.get("score"))