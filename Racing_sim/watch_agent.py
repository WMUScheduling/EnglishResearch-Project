"""Watch a saved PPO racing model drive one or more episodes."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from racing_env import RacingEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved PPO model zip file")
    parser.add_argument("--track", choices=("oval", "technical", "complex"), default="oval")
    parser.add_argument("--progress-reward-scale", type=float, default=0.04)
    parser.add_argument("--off-track-penalty-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of choosing the policy's most likely action",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = RacingEnv(
        render_mode="human",
        track=args.track,
        progress_reward_scale=args.progress_reward_scale,
        off_track_penalty_scale=args.off_track_penalty_scale,
    )
    model = PPO.load(str(args.model), device="cpu")

    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            terminated = False
            truncated = False
            info = {}

            while not terminated and not truncated and not env.window_closed:
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                observation, _, terminated, truncated, info = env.step(action)
                env.render()

            print(
                f"Episode {episode + 1}: "
                f"progress={info.get('progress_laps', 0.0):.3f} laps, "
                f"departures={info.get('off_track_count', 0)}, "
                f"finished={info.get('finished', False)}, "
                f"average_speed={info.get('average_speed', 0.0):.2f}"
            )
            if env.window_closed:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
