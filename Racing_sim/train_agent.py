from stable_baselines3 import PPO

from racing_env import RacingEnv


env = RacingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    seed=42,
    device="cpu",
    ent_coef=0.01,
)

model.learn(total_timesteps=500_000)
model.save("ppo_racing_v4")
env.close()

print("Training finished. Model saved as ppo_racing.zip")