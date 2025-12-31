import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExtendObservation(gym.ObservationWrapper):
    def __init__(self, env, obs_size):
        super().__init__(env)
        self.obs_size = obs_size
        self.prev_state = np.zeros(
            10
        )  # state when the block was last placed (the height of the stack in each column)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def observation(self, observation):
        basic = observation

        root = self.env.unwrapped
        x = root.x
        y = root.y

        active_tetromino = root.active_tetromino
        id = active_tetromino.id
        # matrix = active_tetromino.matrix

        if self.obs_size == 16:
            obs = np.concatenate([basic, [x, y, id]]).astype(np.float32)
        else:
            obs = np.concatenate([basic, [x, y, id], self.prev_state]).astype(
                np.float32
            )

        obs /= 20

        return obs
