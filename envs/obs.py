import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExtendObervation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

    def observation(self, observation):
        basic = observation

        root = self.env.unwrapped
        x = root.x
        y = root.y

        active_tetromino = root.active_tetromino
        id = active_tetromino.id
        # matrix = active_tetromino.matrix

        obs = np.concatenate([basic, [x, y, id]])

        return obs
