from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True, obs_size=26)
    # model = DQN.load("models/dqn_my_plz")
    model = PPO.load("models_test2/ppo26_30M_S1_23")

    tetris_env.play(env, agent=model, delay=1, episodes=10)
