from stable_baselines3 import PPO

from racing_env import RacingEnv


env = RacingEnv(render_mode="human")
model = PPO.load("ppo_racing_v2.zip")

observation, info = env.reset()
terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated and not env.window_closed:
    action, state = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()

print(f"Episode reward: {total_reward:.2f}")
print(f"Progress: {info['progress_laps']:.2f} laps")
print(f"Departures: {info['off_track_count']}")
print(f"Final speed: {env.car_speed:.2f}")
print(f"Ended by stranded rule: {info.get('stranded', False)}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

env.close()