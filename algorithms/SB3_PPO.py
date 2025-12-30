from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
)

model.learn(total_timesteps=20_000_000, tb_log_name="PPO_50M_NL_A_26")
model.save("models/ppo_50M_NL_A_26")
