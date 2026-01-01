from envs import tetris_env
from stable_baselines3 import DQN
import torch
import gymnasium as gym

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device: ", device)

log_dir = "test"

env = gym.make("tetris_gymnasium/Tetris")
# env = tetris_env.make_env(False)
# print(env.observation_space)
model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.00025,
    exploration_fraction=0.2,
    # device=device,
)
# model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50_000_000, log_interval=1000, tb_log_name="DQN_50M_dict")
model.save("models/dqn_50M_dict")

# del model  # remove to demonstrate saving and loading

# if __name__ == "__main__":
#     # model = DQN.load("dqn_tetris")

#     obs, info = env.reset()
#     while True:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             obs, info = env.reset()
