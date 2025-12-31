from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs_test3"
# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=16), n_envs=8)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
)
# std -> (alife=0.05, game_over=-10)_{a =  d_holes==0 -> +0.a, b = -b * d_holes, c=clear_line^2*1 + 1}
model.learn(total_timesteps=10_000_000, tb_log_name="PPO16_10M_std_5_1.0")
model.save("models_test3/PPO16_10M_std_5_1.0")
