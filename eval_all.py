from cross_atc import CrossATCEnv
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch, Circle, RegularPolygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def evaluate_model(model, n_intruders, agent_velocity, max_dheading, n_episodes=1000):
    env = CrossATCEnv(
        n_intruders=n_intruders,
        agent_velocity=agent_velocity,
        max_dheading=max_dheading
    )
    env.reset(seed=SEED)

    stats = {"green": {"count": 0, "success": 0, "durations": []},
             "orange": {"count": 0, "success": 0, "durations": []},
             "red": {"count": 0, "success": 0, "durations": []}}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, min_dist, steps = False, np.inf, 0
        reached_goal = False

        while not done:
            own_pos = env.own_pos.copy()
            for intr in env.intruders:
                dist = np.linalg.norm(own_pos - intr["pos"])
                min_dist = min(min_dist, dist)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            steps += 1

            if reward > 100:
                reached_goal = True

        flag = "red" if min_dist < 1.0 else "orange" if min_dist < 5.0 else "green"
        stats[flag]["count"] += 1
        if reached_goal:
            stats[flag]["success"] += 1
            stats[flag]["durations"].append(steps)

    print(f"{'Category':<10} | {'Count (%)':<18} | {'Successful (%)':<21} | {'Avg duration (steps)':<25}")
    print("-" * 80)
    for cat in ["green", "orange", "red"]:
        c, s, d = stats[cat]["count"], stats[cat]["success"], stats[cat]["durations"]
        p_count = 100 * c / n_episodes
        p_success = 100 * s / c if c > 0 else 0
        avg_dur = np.mean(d) if d else 0
        print(f"{cat:<10} | {c:>4} ({p_count:>5.1f}%)      | {s:>4} ({p_success:>5.1f}%)          | {avg_dur:>7.1f}")
    print("-" * 80)
    total_success = sum(v["success"] for v in stats.values())
    total_duration = sum(sum(v["durations"]) for v in stats.values())
    total_success_episodes = sum(len(v["durations"]) for v in stats.values())
    total_dur_avg = total_duration / total_success_episodes if total_success_episodes > 0 else 0
    print(f"{'Total':<10} | {n_episodes:>4} (100.0%)      | {total_success:>4} ({100*total_success/n_episodes:>5.1f}%)          | {total_dur_avg:>7.1f}\n")

def plot_trajectories(model, n_intruders, agent_velocity, max_dheading, n_episodes=100):
    fig, ax = plt.subplots(figsize=(12, 8))
    safety_flags = []

    for ep in range(n_episodes):
        # env = CrossATCEnv(n_intruders=n_intruders)

        env = CrossATCEnv(
            n_intruders=n_intruders,
            agent_velocity=agent_velocity,
            max_dheading=max_dheading
        )

        obs, _ = env.reset()
        own_traj = []
        intr_start_positions = [intr["pos"].copy() for intr in env.intruders]
        done, min_dist = False, np.inf

        intr_trajs = [[] for _ in range(n_intruders)]

        while not done:
            own_pos = env.own_pos.copy()
            own_traj.append(own_pos)
            for i, intr in enumerate(env.intruders):
                intr_trajs[i].append(intr["pos"].copy())  # ← Record intruder position here
                dist = np.linalg.norm(own_pos - intr["pos"])
                min_dist = min(min_dist, dist)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        # Check safety
        flag = "red" if min_dist < 1.0 else "orange" if min_dist < 5.0 else "green"
        safety_flags.append(flag)

        # Opacity based on goal reached
        traj_alpha = 0.15 if env.reached_goal else 0.0 #voor nu alleen de gelukte zien

        # Mark intruder starting points
        for pos in intr_start_positions:
            marker = plt.Circle(pos, radius=0.3, color=flag, alpha=0.5) #was 0.8
            ax.add_patch(marker)

        # Ownship start triangle
        triangle = RegularPolygon(
            (own_traj[0][0], own_traj[0][1]),
            numVertices=3,
            radius=2,
            orientation=-0.5* np.pi,
            color='blue',
            alpha=0.8 if ep == 0 else 0.3,
        )
        ax.add_patch(triangle)

        # Plot ownship trajectory
        x, y = zip(*own_traj)
        plt.plot(x, y, color=flag, alpha=traj_alpha)

        for traj in intr_trajs:
            if len(traj) > 1:
                x_i, y_i = zip(*traj)
                plt.plot(x_i, y_i, color='gray', linestyle='-', alpha=0.1) #was 0.1

    # Goal circle
    goal_circle = plt.Circle(env.goal, 
                             radius=5.0, 
                             color='green', 
                             linestyle='--',
                             fill=False, 
                             )
    ax.add_patch(goal_circle)

    plt.xlabel("X-position", fontsize=16)
    plt.ylabel("Y-position", fontsize=16)
    ax.set_xlim(-25, 60)
    ax.set_ylim(-40, 40)

    # maak tick labels groter voor latex
    ax.tick_params(axis='both', labelsize=16)

    # custom legend 
    legend_elements = [
        Circle((0, 0), radius=0.5, color='green', label='Safe'),
        Circle((0, 0), radius=0.5, color='orange', label='Warning'),
        # Circle((0, 0), radius=0.5, color='red', label='Critical'),
        RegularPolygon((0, 0), numVertices=3, radius=1.0, orientation=-0.5*np.pi, color='blue', label='Start agent'),
        Rectangle((0, 0), width=2, height=2, fill=False, edgecolor='green', linestyle='--', label='Goal (r=5)'),
        Line2D([0], [0], color='gray', linestyle='-', alpha=0.30, label='Intruder trajectory')
    ]
    ax.legend(handles=legend_elements, fontsize=14)

    ax.set_aspect('equal')
    # plt.title(f"Trajectories over {n_episodes} episodes ({n_intruders} intruder(s))", fontsize=16)
    plt.grid(True)
    plt.show()


def animate_failed_episode(model_path, n_intruders):
    model = DQN.load(model_path)
    env = CrossATCEnv(n_intruders=n_intruders)

    # Try episodes until we find a failed one
    for _ in range(100):  # Max 100 attempts
        obs, _ = env.reset()
        own_traj = []
        intr_traj = {i: [] for i in range(len(env.intruders))}
        done = False

        while not done:
            own_traj.append(env.own_pos.copy())
            for i, intr in enumerate(env.intruders):
                intr_traj[i].append(intr["pos"].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))

        if not env.reached_goal:
            break
    else:
        print("✅ All episodes reached the goal, no animation created.")
        return

    # Prep trajectory data
    own_x, own_y = zip(*own_traj)
    intr_x = {i: [pos[0] for pos in traj] for i, traj in intr_traj.items()}
    intr_y = {i: [pos[1] for pos in traj] for i, traj in intr_traj.items()}

    # Create animation
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-25, 80)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(True)

    line_own, = ax.plot([], [], 'o-', label='Ownship')
    intr_circles = [plt.Circle((0, 0), radius=1.0, color='red', label=f'Intruder {i}') for i in intr_traj]
    for circ in intr_circles:
        ax.add_patch(circ)

    # Goal
    global goal_circle
    goal_circle = patches.Circle(env.goal, radius=5, fill=False, linestyle='--', edgecolor='green', label='Goal (r=5)')
    ax.add_patch(goal_circle)

    # Dynamic safety zone
    dynamic_circle = patches.Circle((own_x[0], own_y[0]), radius=5, fill=False, linestyle='--', edgecolor='blue', label='Ownship zone (r=5)')
    ax.add_patch(dynamic_circle)

    ax.legend()

    def init():
        line_own.set_data([], [])
        for circ in intr_circles:
            circ.center = (-1000, -1000)
        goal_circle.center = env.goal
        dynamic_circle.center = (own_x[0], own_y[0])
        return [line_own] + intr_circles + [goal_circle, dynamic_circle]

    def update(frame):
        line_own.set_data(own_x[:frame], own_y[:frame])
        for i, circ in enumerate(intr_circles):
            circ.center = (intr_x[i][frame - 1], intr_y[i][frame - 1]) if frame > 0 else (intr_x[i][0], intr_y[i][0])
        dynamic_circle.center = (own_x[frame - 1], own_y[frame - 1]) if frame > 0 else (own_x[0], own_y[0])
        return [line_own] + intr_circles + [goal_circle, dynamic_circle]

    ani = animation.FuncAnimation(fig, update, frames=len(own_x), init_func=init, blit=True)
    plt.title("Failed episode animation")
    plt.show()


if __name__ == "__main__":
#     for n in [1,2,3,4]:
#         model = DQN.load(f"saved_models/dqn_cross_atc_{n}intruders")
#         # animate_failed_episode(f"saved_models/dqn_cross_atc_{n}intruders", n)
#         evaluate_model(model, n_intruders=n)
#         plot_trajectories(model, n_intruders=n)

    velocities_and_headings = [
        (1.5, 0.3),
        # (1.4, 0.28),
        # (1.3, 0.26),
        # (1.2, 0.24),
        # (1.1, 0.22)
    ]

    for vel, dh in velocities_and_headings:
        model_name = f"saved_models/dqn_cross_atc_1intruder_v{vel:.1f}_dh{dh:.2f}"
        print(f"\n🔍 Evaluating model: {model_name}")
        model = DQN.load(model_name)
        evaluate_model(model, n_intruders=1, agent_velocity=vel, max_dheading=dh)
        plot_trajectories(model, n_intruders=1, agent_velocity=vel, max_dheading=dh)