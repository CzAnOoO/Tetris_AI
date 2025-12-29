import gymnasium as gym
from tetris_gymnasium.mappings.rewards import RewardsMapping


# https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper.env
class MyReward(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        """
        https://github.com/Max-We/Tetris-Gymnasium/blob/main/tetris_gymnasium/wrappers/observation.py#L114
        observation space is 1D vector "obs": 
        - The height of each column: obs[0...9]
        - The maximum height : obs[10]
        - The number of holes : obs[11]
        - The bumpiness : obs[12]
        """
        max_height = obs[10]
        holes = obs[11]
        bumpiness = obs[12]

        if reward != 0:
            reward += -0.05 * max_height - 0.05 * holes - 0.05 * bumpiness

        return obs, reward, term, trun, info
