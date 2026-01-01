from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs_test5"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=16), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
)
# (alife=0.05, game_over=-10)
# PPO{obs_size}_{T-steps}_{d_holes==0 -> +0.a}_{-0.b * d_holes} _{clear_line^c * 1 + 1}_{-0.0d * d_height}_{-0.0e * d_bump}
model.learn(total_timesteps=100_000, tb_log_name="PPO_50M_8_5_2_1_1_")
model.save("models_test5/PPO_50M_02_5_2_1_1")
