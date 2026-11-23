"""Open a static preview of a racing track without training an agent."""

import argparse

from racing_env import RacingEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=("oval", "technical", "complex"),
        default="complex",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    env = RacingEnv(render_mode="human", track=args.track)
    env.reset(seed=args.seed)
    env.render()
    print(f"Previewing the {args.track} track. Close the window to exit.")

    try:
        while not env.window_closed:
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
