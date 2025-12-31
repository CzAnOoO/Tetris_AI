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
        id_index = active_tetromino.id  # 2 - 8 = 7 st
        # matrix = active_tetromino.matrix
        id = np.zeros(7)
        id[id_index - 2]

        xy = [x / 10.0, y / 20.0]
        id_val = [(id_index - 2) / 6]
        if self.obs_size == 22:
            pieces = [basic / 20.0, xy, id]
        elif self.obs_size == 16:
            pieces = [basic / 20.0, xy, id_val]
        elif self.obs_size == 32:
            pieces = [basic / 20.0, xy, id, self.prev_state / 20]
        else:
            pieces = [basic / 20.0, xy, id_val, self.prev_state / 20]

        obs = np.concatenate(pieces).astype(np.float32)

        return obs
