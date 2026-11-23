"""Drive a saved PPO policy on screen."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from racing_env import RacingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("oval", "technical"), default="technical")
    parser.add_argument("--progress-reward-scale", type=float, default=0.04)
    parser.add_argument("--off-track-penalty-scale", type=float, default=1.0)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    env = RacingEnv(render_mode="human", track=args.track, progress_reward_scale=args.progress_reward_scale, off_track_penalty_scale=args.off_track_penalty_scale)
    model = PPO.load(str(args.model), device="cpu")
    observation, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    info = {}

    while not terminated and not truncated and not env.window_closed:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        env.render()

    print(f"Episode reward: {total_reward:.2f}")
    print(f"Progress: {info.get('progress_laps', 0.0):.2f} laps")
    print(f"Departures: {info.get('off_track_count', 0)}")
    print(f"Finished: {info.get('finished', False)}")
    print(f"Stranded: {info.get('stranded', False)}")
    env.close()


if __name__ == "__main__":
    main()
