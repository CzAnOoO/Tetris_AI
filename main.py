from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True, obs_size=16)
    # model = DQN.load("models/dqn_my_plz")
    model = PPO.load("models/ppo_20M")

    tetris_env.play(env, agent=model, delay=10, episodes=10)
