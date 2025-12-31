import cv2
from envs import tetris_env

if __name__ == "__main__":
    env = tetris_env.make_env(True, 42, obs_size=16)

    terminated = False
    last_score = 0
    while not terminated:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100)  # timeout to see the movement
        # print(info.get("lines_cleared", 0))
        if reward != 0:
            print(reward)
        # print(observation[15])
        current_score = info.get("lines_cleared", 0)
        if current_score != last_score:
            print("Score:", current_score)
            last_score = current_score
    print("Game Over!")
