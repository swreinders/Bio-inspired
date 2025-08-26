import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
from cross_atc import CrossATCEnv

SEED = 42

# define the six configurations for sensan
tuning_configs = [
    {"gamma": 0.8, "lr": 0.001, "eps_start": 0.3, "eps_end": 0.1}, 
    {"gamma": 1.0, "lr": 0.001, "eps_start": 0.3, "eps_end": 0.1},
    {"gamma": 0.9, "lr": 0.002, "eps_start": 0.3, "eps_end": 0.1},
    {"gamma": 0.9, "lr": 0.0005, "eps_start": 0.3, "eps_end": 0.1},
    {"gamma": 0.9, "lr": 0.001, "eps_start": 0.4, "eps_end": 0.2},
    {"gamma": 0.9, "lr": 0.001, "eps_start": 0.2, "eps_end": 0.0},
]
# keeping track of model performance during training
def train_with_checkpoints(config, config_id, total_timesteps=300_000, save_every=1000):
    env = CrossATCEnv(
        n_intruders=1,
        agent_velocity=1.5,
        max_dheading=0.3
    )

    model = DQN(
        "MlpPolicy",
        env,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        train_freq=(1, "step"),
        exploration_initial_eps=config["eps_start"],
        exploration_final_eps=config["eps_end"],
        exploration_fraction=0.9,
        verbose=1,
        seed=SEED
    )

    save_dir = f"sensitivity_models/config_{config_id}"
    os.makedirs(save_dir, exist_ok=True)

    for step in range(0, total_timesteps, save_every):
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        checkpoint_path = os.path.join(save_dir, f"model_step_{step + save_every}.zip")
        model.save(checkpoint_path)
        print(f" Saved checkpoint for config {config_id} at step {step + save_every}")

def main():
    for i, config in enumerate(tuning_configs):
        print(f"\nStarting training for config {i+1}: {config}")
        train_with_checkpoints(config, config_id=i+1)


def evaluate_model_return_stats(model, n_intruders, agent_velocity, max_dheading, n_episodes=100):
    env = CrossATCEnv(n_intruders=n_intruders, agent_velocity=agent_velocity, max_dheading=max_dheading)
    env.reset(seed=SEED)

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

def analyze_config(config_name):
    csv_path = f"{config_name}_analysis.csv"

    if os.path.exists(csv_path):
        print(f"Loading existing analysis from {csv_path}")
        df = pd.read_csv(csv_path)
        valid_steps = df["step"].tolist()
    else:
        print(f"⚙️ Running fresh analysis for {config_name}")
        model_dir = f"sensitivity_models/{config_name}"
        timesteps = list(range(1000, 305000, 5000))
        results, valid_steps = [], []

        for step in timesteps:
            model_path = os.path.join(model_dir, f"model_step_{step}.zip")
            if not os.path.exists(model_path):
                continue
            print(f"🔍 Analyzing {config_name} step {step}")
            model = DQN.load(model_path)
            stats = evaluate_model_return_stats(model, n_intruders=1, agent_velocity=1.1, max_dheading=0.22, n_episodes=100)

            total = sum(v["count"] for v in stats.values())
            if total == 0:
                continue

            valid_steps.append(step)
            results.append({"step": step, **{f"{k}_{m}": v[m] for k, v in stats.items() for m in ["count", "success"]}})

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    # safety proportions
    green_props = df["green_count"] / (df["green_count"] + df["orange_count"] + df["red_count"])
    orange_props = df["orange_count"] / (df["green_count"] + df["orange_count"] + df["red_count"])
    red_props = df["red_count"] / (df["green_count"] + df["orange_count"] + df["red_count"])

    # total success rate
    total_success_rate = (
        df["green_success"] + df["orange_success"] + df["red_success"]
    ) / (df["green_count"] + df["orange_count"] + df["red_count"] + 1e-6)

    #plot proportions and total success
    plt.figure(figsize=(14, 6))
    plt.plot(valid_steps, smooth(green_props), label='Safe', color='green', linewidth=3)
    plt.plot(valid_steps, smooth(orange_props), label='Warning', color='orange', linewidth=3)
    plt.plot(valid_steps, smooth(red_props), label='Critical', color='red', linewidth=3)
    plt.plot(valid_steps, smooth(total_success_rate), label='Total Success Rate', color='black', linewidth=3)

    plt.xlabel("Training Steps", fontsize=20)
    plt.ylabel("Proportion", fontsize=20)
    plt.title("Trajectory Category Distribution + Total Success Rate", fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # main()
    analyze_config("config_1")
    # analyze_config("config_2") 
    # analyze_config("config_3")
    # analyze_config("config_4")
    # analyze_config("config_5")
    # analyze_config("config_6")

