from envs import tetris_env
from stable_baselines3 import DQN
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = tetris_env.make_env(True, obs_size=16)
    # model = DQN.load("models/dqn_my_plz")
    model = PPO.load("models/ppo_50M_L")

    tetris_env.play(env, agent=model, delay=5, episodes=20)
