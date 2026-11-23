"""Continue a baseline PPO checkpoint for a timed practice condition."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from racing_env import RacingEnv


class PracticeTimeLimit(BaseCallback):
    def __init__(self, hours, verbose=0):
        super().__init__(verbose)
        self.budget_seconds = hours * 3600.0
        self.started_at = None

    def _on_training_start(self):
        self.started_at = time.monotonic()

    def _on_step(self):
        return time.monotonic() - self.started_at < self.budget_seconds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--practice-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.practice_hours <= 0:
        raise SystemExit("practice-hours must be greater than zero")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("training_runs") / f"practice_{args.label}_{stamp}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        RacingEnv(track="technical"),
        filename=str(run_dir / "train"),
        info_keywords=(
            "progress_laps",
            "off_track_count",
            "finished",
            "stranded",
            "episode_steps",
        ),
    )
    model = PPO.load(str(args.baseline_model), env=env, device="cpu")
    model.set_random_seed(args.seed)
    model.tensorboard_log = str(run_dir / "tensorboard")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="practice_checkpoint",
        verbose=1,
    )
    time_limit = PracticeTimeLimit(args.practice_hours, verbose=1)
    starting_steps = model.num_timesteps
    started_at = time.monotonic()

    model.learn(
        total_timesteps=1_000_000_000,
        callback=[checkpoint_callback, time_limit],
        reset_num_timesteps=False,
        tb_log_name="PPO_practice",
    )
    elapsed_seconds = time.monotonic() - started_at
    additional_steps = model.num_timesteps - starting_steps
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    env.close()

    config = {
        "stage": "technical_track_practice",
        "track": "technical",
        "baseline_model": str(args.baseline_model),
        "seed": args.seed,
        "label": args.label,
        "requested_practice_hours": args.practice_hours,
        "actual_elapsed_seconds": round(elapsed_seconds, 2),
        "additional_environment_steps": int(additional_steps),
        "checkpoint_frequency": args.checkpoint_freq,
        "device": "cpu",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Practice complete. Run records: {run_dir}")
    print(f"Requested hours: {args.practice_hours:.2f}")
    print(f"Elapsed seconds: {elapsed_seconds:.1f}")
    print(f"Additional environment steps: {additional_steps}")
    print(f"Final model: {final_path}.zip")


if __name__ == "__main__":
    main()
