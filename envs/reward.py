import gymnasium as gym
from tetris_gymnasium.mappings.rewards import RewardsMapping


# https://gymnasium.farama.org/api/wrappers/#methods
class MyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_max_height = None
        self.prev_holes = None
        self.prev_bumpiness = None

    # https://stackoverflow.com/questions/73675262/openai-gym-problem-override-observationwrapper-reset-method
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_max_height = obs[10]
        self.prev_holes = obs[11]
        self.prev_bumpiness = obs[12]
        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        lines_cleared = info.get("lines_cleared", 0)
        if lines_cleared > 0:
            reward += (lines_cleared**1.5) * 50

        """
        https://github.com/Max-We/Tetris-Gymnasium/blob/main/tetris_gymnasium/wrappers/observation.py#L114
        observation space is 1D vector "obs": 
        - The height of each column: obs[0...9]
        - The maximum height : obs[10]
        - The number of holes : obs[11]
        - The bumpiness : obs[12]
        """

        if reward != 0:
            max_height = obs[10]
            holes = obs[11]
            bumpiness = obs[12]
            # delta
            d_height = max_height - self.prev_max_height
            d_holes = holes - self.prev_holes
            d_bump = bumpiness - self.prev_bumpiness

            reward += -0.05 * d_height - 1.5 * d_holes - 0.01 * d_bump
            self.prev_max_height = max_height
            self.prev_holes = holes
            self.prev_bumpiness = bumpiness
            if d_holes == 0:
                reward += 30
                # print("0 delta hole")

        if action == 6:  # swap
            reward -= 0.5
        if action == 7:  # no action
            reward += 0.02

        return obs, reward, term, trun, info
