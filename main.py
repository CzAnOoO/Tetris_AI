from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True)
    model = DQN.load("models/dqn_1")
    # model = PPO.load("models/ppo_1")

    tetris_env.play(env, model, 100)
