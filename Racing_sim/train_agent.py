"""Train PPO and save checkpoints, evaluation results, and TensorBoard logs."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from racing_env import RacingEnv


EPISODE_INFO = (
    "progress_laps",
    "off_track_count",
    "finished",
    "stranded",
    "episode_steps",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("training_runs") / f"ppo_{stamp}_seed{args.seed}"
    model_dir = run_dir / "models"
    checkpoint_dir = model_dir / "checkpoints"
    best_dir = model_dir / "best"
    evaluation_dir = run_dir / "evaluation"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "track": "technical_multiturn_v2",
        "track_control_points": int(len(RacingEnv.CONTROL_POINTS)),
        "track_samples_per_segment": 8,
        "track_half_width_pixels": 22,
        "reset_lateral_offset_pixels": [-8, 8],
        "reset_heading_jitter_degrees": [-5, 5],
        "reset_speed_range": [0, 0.4],
        "action_space": "MultiDiscrete([3, 5])",
        "observation_features": [
            "heading_error_sin",
            "heading_error_cos",
            "normalized_lateral_offset",
            "normalized_speed",
            "lookahead_55_forward",
            "lookahead_55_lateral",
            "lookahead_125_forward",
            "lookahead_125_lateral",
            "lap_progress",
            "departure_fraction",
        ],
        "total_timesteps": args.timesteps,
        "seed": args.seed,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "policy_network": [128, 128],
        "evaluation_frequency": args.eval_freq,
        "checkpoint_frequency": args.checkpoint_freq,
        "evaluation_episodes": 5,
        "device": "cpu",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    train_env = Monitor(
        RacingEnv(),
        filename=str(run_dir / "train"),
        info_keywords=EPISODE_INFO,
    )
    eval_env = Monitor(
        RacingEnv(),
        filename=str(run_dir / "evaluation" / "episodes"),
        info_keywords=EPISODE_INFO,
    )
    eval_env.reset(seed=args.seed + 10_000)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_racing",
        verbose=1,
    )
    evaluation_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(evaluation_dir),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, evaluation_callback],
        tb_log_name="PPO",
    )
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    train_env.close()
    eval_env.close()

    print(f"Training complete. Run records: {run_dir}")
    print(f"Final model: {final_path}.zip")
    print(f"Best evaluated model: {best_dir / 'best_model.zip'}")
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
