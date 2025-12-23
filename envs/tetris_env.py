import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris


def make_env(render=False, seed=None):
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
    return env


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

        key = cv2.waitKey(100)  # timeout to see the movement
        current_score = info.get("lines_cleared", 0)
        if current_score != last_score:
            print("Score:", current_score)
            last_score = current_score
    print("Game Over!")
