# cross_atc.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CrossATCEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, d=20.0, d_goal=40.0, n_intruders=3, agent_velocity=1.5, max_dheading=0.3):
        super().__init__()
        self.d = d # d: the half-distance from the origin where intruders start
        self.d_goal = d_goal # d_goal: X-coordinate of your agents goal
        self.n_intruders = n_intruders
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3+3*n_intruders,), dtype=float) 
        self.action_space = spaces.Discrete(9)  # (steep) left, straight, (steep) right
        self.step_count = 0 # initialize step counterj
        self.reached_goal = False
        self.agent_velocity = agent_velocity
        self.max_dheading = max_dheading

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # agent begint op -d van origin
        self.own_pos     = np.array([-self.d, 0.0])
        self.own_heading = self.np_random.uniform(-0.0, 0.0)

        # Doel ligt op 2d van origin
        self.goal        = np.array([2*self.d, 0.0])
        self.reached_goal = False

        # Intruders spawnen random in een 10×10 gebied rond origin
        self.intruders = []
        for _ in range(self.n_intruders):
            x = self.np_random.uniform(-5, 5)
            y = self.np_random.uniform(-5, 5)
            pos = np.array([x, y])
            heading_to_ownship = np.arctan2(self.own_pos[1] - y, self.own_pos[0] - x)
            self.intruders.append({"pos": pos, "heading": heading_to_ownship})

        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self): #obstervation
        # Start with ownship → goal
        delta_goal = self.goal - self.own_pos # vector to goal
        dist_goal  = np.linalg.norm(delta_goal) # distance to goal
        angle_goal = np.arctan2(delta_goal[1], delta_goal[0]) - self.own_heading # angle to goal
        obs = [dist_goal, angle_goal, self.own_heading]  # afstand en hoek naar doel, plus huidige heading

        # Then each intruder: distance, bearing, rel-heading
        for intr in self.intruders:
            delta_intr = intr["pos"] - self.own_pos # distance to intruders
            obs += [np.linalg.norm(delta_intr), np.arctan2(delta_intr[1], delta_intr[0]) - self.own_heading, intr["heading"] - self.own_heading]
            # obs wordt aangevuld met afstand tot intr, hoek van intr en relatieve hoek van agent en intr
        return np.array(obs, dtype=float)

    def step(self, action):
        prev_dist_goal = np.linalg.norm(self.goal - self.own_pos)
        action = int(action)
        delta_heading = {
            0: -self.max_dheading,
            1: -0.20,
            2: -0.10,
            3: -0.05,
            4:  0.0,
            5:  0.05,
            6:  0.10,
            7:  0.20,
            8:  self.max_dheading
        }[action]
        self.own_heading += delta_heading
        self.own_pos += self.agent_velocity * np.array([
            np.cos(self.own_heading),
            np.sin(self.own_heading)
        ])

        for intr in self.intruders:
            # Update heading towards current ownship position
            delta = self.own_pos - intr["pos"]
            intr["heading"] = np.arctan2(delta[1], delta[0])

            # Move in that direction
            intr["pos"] += np.array([
                np.cos(intr["heading"]),
                np.sin(intr["heading"])
            ])

        # lichte straf per stap + straf voor draaien
        reward = 0.0
        done = False

        # voortgang richting doel (geclipped voor stabiliteit)
        new_dist_goal = np.linalg.norm(self.goal - self.own_pos)
        progress = np.clip(prev_dist_goal - new_dist_goal, -1.0, 1.0)
        reward += 60.0 * progress  

        #  beloon finish
        if new_dist_goal < 5.0:
            reward += 1000.0
            done = True
            self.reached_goal = True

        for intr in self.intruders:
            dist = np.linalg.norm(intr["pos"] - self.own_pos)
            if dist < 10.0:
                penalty = (10.0 - dist)**4  * 0.5 
                reward -= penalty
        
        self.step_count += 1
        if self.step_count >= 400: 
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        # potentiele console‐weergave van posities
        print(f"Own: {self.own_pos}, Goal: {self.goal}")
        for i, intr in enumerate(self.intruders):
            print(f"Intr{i}: {intr['pos']}")