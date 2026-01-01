from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=16), n_envs=8)

model = PPO.load("models_con/PPO_8_5_2_1_1", vec_env)
# (alife=0.05, game_over=-10)
# PPO{obs_size}_{T-steps}_{d_holes==0 -> +0.a}_{-0.b * d_holes} _{clear_line^c * 1 + 1}_{-0.d * d_height}
model.learn(
    total_timesteps=5_000_000,
    tb_log_name="PPO_8_5_2_1_1",
    reset_num_timesteps=False,
)
model.save("models_con/PPO_8_5_2_1_1")

# uv run python -m algorithms.SB3_PPO_con
