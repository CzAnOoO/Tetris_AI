from envs import tetris_env
from stable_baselines3 import DQN

env = tetris_env.make_env(False)
# print(env.observation_space)
model = DQN("MlpPolicy", env, verbose=1)
# model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=5000000, log_interval=1000)
model.save("models/dqn_my_r2")

# del model  # remove to demonstrate saving and loading

# if __name__ == "__main__":
#     # model = DQN.load("dqn_tetris")

#     obs, info = env.reset()
#     while True:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             obs, info = env.reset()
