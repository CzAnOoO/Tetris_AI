from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True)
    # model = DQN.load("models/dqn_my_plz")
    model = PPO.load("models/ppo_plz_50M")

    tetris_env.play(env, model, 100)
