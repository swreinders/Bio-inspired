import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
from cross_atc import CrossATCEnv

SEED = 42

def train_with_checkpoints(n_intruders, agent_velocity, max_dheading, total_timesteps, save_every):
    env = CrossATCEnv(
        n_intruders=n_intruders,
        agent_velocity=agent_velocity,
        max_dheading=max_dheading
    )
    model = DQN(
        "MlpPolicy",
        env,
        gamma=0.9,
        learning_rate=1e-3,
        train_freq=(1, "step"),
        exploration_initial_eps=0.3,
        exploration_final_eps=0.1,
        exploration_fraction=0.9,
        verbose=1,
    )

    save_dir = "analyzed_models"
    os.makedirs(save_dir, exist_ok=True)

    for step in range(0, total_timesteps, save_every):
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        checkpoint_path = os.path.join(save_dir, f"MAINMODEL_STEP{step + save_every}.zip")
        model.save(checkpoint_path)
        print(f"Saved checkpoint at step {step + save_every}")

def evaluate_model_return_stats(model, n_intruders, agent_velocity, max_dheading, n_episodes=100):
    env = CrossATCEnv(
        n_intruders=n_intruders,
        agent_velocity=agent_velocity,
        max_dheading=max_dheading
    )
    obs, _ = env.reset(seed=SEED)

    stats = {"green": {"count": 0, "success": 0},
             "orange": {"count": 0, "success": 0},
             "red": {"count": 0, "success": 0}}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, min_dist, reached_goal = False, np.inf, False

        while not done:
            own_pos = env.own_pos.copy()
            for intr in env.intruders:
                dist = np.linalg.norm(own_pos - intr["pos"])
                min_dist = min(min_dist, dist)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            if reward > 100:
                reached_goal = True

        flag = "red" if min_dist < 1.0 else "orange" if min_dist < 5.0 else "green"
        stats[flag]["count"] += 1
        if reached_goal:
            stats[flag]["success"] += 1

    return stats

def smooth(y, box_pts=5):
    return np.convolve(y, np.ones(box_pts)/box_pts, mode='same')

def analyze_mainmodel_progress():
    timesteps = list(range(0, 305000, 5000))
    green_props, orange_props, red_props = [], [], []
    valid_steps = []
    results = []

    for step in timesteps:
        model_path = f"analyzed_models/MAINMODEL_STEP{step}.zip"
        if not os.path.exists(model_path):
            continue
        print(f"Analyzing MAINMODEL_STEP{step}")
        model = DQN.load(model_path)
        stats = evaluate_model_return_stats(model, n_intruders=2, agent_velocity=1.5, max_dheading=0.3, n_episodes=100)

        total = sum(v["count"] for v in stats.values())
        if total == 0:
            continue

        valid_steps.append(step)
        results.append({
            "timestep": step,
            "green_count": stats["green"]["count"],
            "green_success": stats["green"]["success"],
            "orange_count": stats["orange"]["count"],
            "orange_success": stats["orange"]["success"],
            "red_count": stats["red"]["count"],
            "red_success": stats["red"]["success"],
        })

        green_props.append(stats["green"]["count"] / total)
        orange_props.append(stats["orange"]["count"] / total)
        red_props.append(stats["red"]["count"] / total)

    df = pd.DataFrame(results)

    # Total success rate
    total_success = (
        df["green_success"] + df["orange_success"] + df["red_success"]
    ) / (df["green_count"] + df["orange_count"] + df["red_count"] + 1e-6)

    # Plot combined graph
    plt.figure(figsize=(14, 6))
    plt.plot(valid_steps, smooth(green_props), label='Safe', color='green', linewidth=3)
    plt.plot(valid_steps, smooth(orange_props), label='Warning', color='orange', linewidth=3)
    plt.plot(valid_steps, smooth(red_props), label='Critical', color='red', linewidth=3)
    plt.plot(valid_steps, smooth(total_success), label='Total Success Rate', color='black', linewidth=3)

    plt.xlabel("Training Steps", fontsize=20)
    plt.ylabel("Proportion", fontsize=20)
    plt.title("Trajectory Category Distribution + Total Success Rate", fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df.to_csv("model_progress.csv", index=False)
    print("Saved model progress to model_progress.csv")

if __name__ == "__main__":

    # train_with_checkpoints(
    #     n_intruders=2,
    #     agent_velocity=1.5,
    #     max_dheading=0.3,
    #     total_timesteps=300_000,
    #     save_every=5000
    # )

    analyze_mainmodel_progress()
