"""Evaluate a saved racing policy over reproducible episode seeds."""

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from racing_env import RacingEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    env = RacingEnv()
    model = PPO.load(str(args.model), device="cpu")
    records = []

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        observation, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        on_track_steps = 0
        info = {}

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            on_track_steps += int(info["on_track"])

        steps = int(info.get("episode_steps", 0))
        records.append(
            {
                "episode": episode + 1,
                "seed": episode_seed,
                "return": round(total_reward, 4),
                "progress_laps": round(float(info.get("progress_laps", 0.0)), 4),
                "finished": bool(info.get("finished", False)),
                "departures": int(info.get("off_track_count", 0)),
                "stranded": bool(info.get("stranded", False)),
                "steps": steps,
                "on_track_fraction": round(on_track_steps / max(steps, 1), 4),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Episodes: {len(records)}")
    print(f"Completion rate: {np.mean([r['finished'] for r in records]):.2%}")
    print(f"Mean laps: {np.mean([r['progress_laps'] for r in records]):.3f}")
    print(f"Mean departures: {np.mean([r['departures'] for r in records]):.2f}")
    print(f"Mean on-track fraction: {np.mean([r['on_track_fraction'] for r in records]):.2%}")
    print(f"Episode results: {args.output}")
    env.close()


if __name__ == "__main__":
    main()
