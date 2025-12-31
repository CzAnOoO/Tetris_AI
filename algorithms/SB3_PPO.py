from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs_test2"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=26), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
)

model.learn(total_timesteps=10_000_000, tb_log_name="PPO26_10M_S13_23")
model.save("models_test2/ppo26_10M_S13_23")
