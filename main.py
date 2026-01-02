from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True, obs_size=22)
    # model = DQN.load("models/dqn_my_plz")
    # PPO16_10M_01_5_2_1
    model = PPO.load("models_con/PPO22_8_5_2_0_5_1")

    tetris_env.play(env, agent=model, delay=1, episodes=10, render=True)
