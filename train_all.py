from cross_atc import CrossATCEnv
from stable_baselines3 import DQN
import gymnasium as gym
import os

def train_model(n_intruders, agent_velocity, max_dheading, timesteps=250000): #op 250000 houden
    env_id = f"CrossATC-{n_intruders}intruders-v0"
    gym.envs.registration.register(
        id=env_id,
        entry_point="cross_atc:CrossATCEnv",
        kwargs={
            "n_intruders": n_intruders,
            "agent_velocity": agent_velocity,
            "max_dheading": max_dheading
        }
    )
    env = gym.make(env_id)

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
    model.learn(total_timesteps=timesteps)

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

    fname = f"dqn_cross_atc_1intruder_v{agent_velocity:.1f}_dh{max_dheading:.2f}"
    model.save(os.path.join(save_dir, fname))
    print(f"Trained and saved model with {n_intruders} intruder(s)")

if __name__ == "__main__":
    # for n in [3]:
    #     train_model(n_intruders=n)

    # alle configuraties voor speed ratios
    settings = [
        # (1.5, 0.30),
        (1.4, 0.28),
        (1.3, 0.26),
        (1.2, 0.24),
        (1.1, 0.22)
    ]

    for velocity, dheading in settings:
        train_model(n_intruders=1, agent_velocity=velocity, max_dheading=dheading)

