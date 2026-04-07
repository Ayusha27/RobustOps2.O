from env.environment import RobustOpsEnv
from env.models import Action
import random
import matplotlib.pyplot as plt


# ---- Agents ---- #

def baseline_agent(signals):
    if "suspicious_domain" in signals or "spoofed_sender" in signals:
        return "spam"
    return "not_spam"


def improved_agent(signals):
    strong = ["suspicious_domain", "spoofed_sender"]
    count = sum(1 for s in signals if s in strong)

    if count >= 1:
        return "spam"
    elif "urgent_tone" in signals:
        return None  # uncertain
    else:
        return "not_spam"


# ---- Runner ---- #

def run_episode(env, agent_fn):
    obs = env.reset()

    # Extract signals from message
    # message = "... | Signals: ['urgent_tone', ...]"
    signals_str = obs.message.split("Signals: ")[-1]
    signals = eval(signals_str)

    # Step 1: classify
    decision = agent_fn(signals)

    if decision is None:
        action = Action(action_type="flag_uncertain", content=None)
    else:
        action = Action(action_type="classify", content=decision)

    obs, reward1, done, _ = env.step(action)

    # Step 2: revise (only if not done)
    if not done:
        decision = agent_fn(signals)

        if decision is None:
            action = Action(action_type="flag_uncertain", content=None)
        else:
            action = Action(action_type="revise", content=decision)

        obs, reward2, done, _ = env.step(action)
    else:
        reward2 = type(reward1)(value=0.0)

    return reward1.value + reward2.value


def evaluate(agent_fn, episodes=50):
    env = RobustOpsEnv()

    rewards = []

    for _ in range(episodes):
        r = run_episode(env, agent_fn)
        rewards.append(r)

    avg_reward = sum(rewards) / len(rewards)
    return avg_reward, rewards

def plot_results(baseline_rewards, improved_rewards):
    plt.figure()

    plt.plot(baseline_rewards, label="Baseline")
    plt.plot(improved_rewards, label="Improved")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Agent Performance Under Noisy Multi-Step Decision Making")

    plt.legend()
    plt.grid()

    plt.show()

# ---- Main ---- #

if __name__ == "__main__":
    baseline_avg, baseline_rewards = evaluate(baseline_agent)
    improved_avg, improved_rewards = evaluate(improved_agent)

    print("\n--- Evaluation Results ---")
    print(f"Baseline Avg Reward: {baseline_avg:.2f}")
    print(f"Improved Avg Reward: {improved_avg:.2f}")

    print("\nSample Rewards:")
    print("Baseline:", baseline_rewards[:5])
    print("Improved:", improved_rewards[:5])
    plot_results(baseline_rewards, improved_rewards)