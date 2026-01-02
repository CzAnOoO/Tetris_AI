from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs_con"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=26), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
    ent_coef=0.01,
)
# (alife=0.05, game_over=-10)
# PPO{obs_size}_{T-steps}_{d_holes==0 -> +0.a}_{-0.b * d_holes} _{clear_line^c * 1 + 1}_{-0.0d * d_height}_{-0.0e * d_bump}_{-0.0f * max_height}
model.learn(total_timesteps=50_000_000, tb_log_name="PPO22_8_5_2_1_3_coef1")
model.save("models_con/PPO22_8_5_2_1_3_coef1")
