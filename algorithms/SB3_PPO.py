from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs_test2"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=16), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
)

model.learn(total_timesteps=30_000_000, tb_log_name="PPO16_30M_S1_23_s")
model.save("models_test2/ppo16_30M_S1_23_s")
