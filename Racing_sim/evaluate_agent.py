"""Evaluate a saved racing policy over reproducible episode seeds."""

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from racing_env import RacingEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("oval", "technical", "complex"), default="technical")
    parser.add_argument("--progress-reward-scale", type=float, default=0.04)
    parser.add_argument("--off-track-penalty-scale", type=float, default=1.0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    env = RacingEnv(track=args.track, progress_reward_scale=args.progress_reward_scale, off_track_penalty_scale=args.off_track_penalty_scale)
    model = PPO.load(str(args.model), device="cpu")
    records = []

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        observation, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        on_track_steps = 0
        speed_sum = 0.0
        low_speed_steps = 0
        info = {}

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            on_track_steps += int(info["on_track"])
            speed_sum += float(info["speed"])
            low_speed_steps += int(info["speed"] < 0.5)

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
                "average_speed": round(speed_sum / max(steps, 1), 4),
                "low_speed_fraction": round(low_speed_steps / max(steps, 1), 4),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Episodes: {len(records)}")
    completion = np.array([r["finished"] for r in records], dtype=float)
    laps = np.array([r["progress_laps"] for r in records], dtype=float)
    departures = np.array([r["departures"] for r in records], dtype=float)
    on_track = np.array([r["on_track_fraction"] for r in records], dtype=float)
    average_speed = np.array([r["average_speed"] for r in records], dtype=float)
    low_speed = np.array([r["low_speed_fraction"] for r in records], dtype=float)
    completed_steps = [r["steps"] for r in records if r["finished"]]

    print(f"Completion rate: {completion.mean():.2%} (SD {completion.std(ddof=1):.2%})")
    print(f"Laps: mean {laps.mean():.3f}, SD {laps.std(ddof=1):.3f}")
    print(f"Departures: mean {departures.mean():.2f}, SD {departures.std(ddof=1):.2f}")
    print(f"On-track fraction: mean {on_track.mean():.2%}, SD {on_track.std(ddof=1):.2%}")
    print(f"Average speed: mean {average_speed.mean():.2f}, SD {average_speed.std(ddof=1):.2f}")
    print(f"Low-speed fraction: mean {low_speed.mean():.2%}, SD {low_speed.std(ddof=1):.2%}")
    if completed_steps:
        print(f"Steps per completed lap: mean {np.mean(completed_steps):.1f}")
    else:
        print("Steps per completed lap: no completed laps")
    print(f"Episode results: {args.output}")
    env.close()


if __name__ == "__main__":
    main()
