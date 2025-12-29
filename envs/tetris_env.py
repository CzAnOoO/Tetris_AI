import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

my_rewards = RewardsMapping(
    alife=0.01, clear_line=20.0, game_over=-1.0, invalid_action=-0.1
)


def make_env(render=False, seed=None):
    env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human" if render else None,
        rewards_mapping=my_rewards,
    )
    env = FeatureVectorObservation(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def play(env, agent=None, delay=100):
    obs, info = env.reset()
    terminated = False
    truncated = False
    last_score = 0

    while not (terminated or truncated):
        env.render()

        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_score = info.get("lines_cleared", 0)
        if current_score != last_score:
            print("Score:", current_score)
            last_score = current_score
        cv2.waitKey(delay)

    print("Game Over!")


if __name__ == "__main__":
    # Step 1
    env = make_env(True, 42)

    terminated = False
    last_score = 0
    while not terminated:
        env.render()
        # Step 2: run a agent
        # action = FROM AGENT
        observation, reward, terminated, truncated, info = env.step(action)

        key = cv2.waitKey(1000)  # timeout to see the movement
        current_score = info.get("lines_cleared", 0)
        if current_score != last_score:
            print("Score:", current_score)
            last_score = current_score
    print("Game Over!")
