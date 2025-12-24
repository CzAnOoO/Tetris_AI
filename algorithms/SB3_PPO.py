from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False), n_envs=8)

model = PPO("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("models/ppo_1")
