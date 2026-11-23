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
    "average_speed",
    "low_speed_fraction",
    "on_track_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("oval", "technical", "complex"), default="technical")
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--progress-reward-scale", type=float, default=0.04)
    parser.add_argument("--off-track-penalty-scale", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("training_runs") / f"ppo_{args.track}_r{args.progress_reward_scale:g}_p{args.off_track_penalty_scale:g}_{stamp}_seed{args.seed}"
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
        "track": args.track,
        "track_control_points": {"oval": 36, "technical": len(RacingEnv.CONTROL_POINTS), "complex": len(RacingEnv.COMPLEX_CONTROL_POINTS)}[args.track],
        "track_samples_per_segment": 12 if args.track == "complex" else 8,
        "track_half_width_pixels": {"oval": 50, "technical": 22, "complex": 18}[args.track],
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
        "progress_reward_scale": args.progress_reward_scale,
        "off_track_penalty_scale": args.off_track_penalty_scale,
        "resume_model": str(args.resume_model) if args.resume_model else None,
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
        RacingEnv(track=args.track, progress_reward_scale=args.progress_reward_scale, off_track_penalty_scale=args.off_track_penalty_scale),
        filename=str(run_dir / "train"),
        info_keywords=EPISODE_INFO,
    )
    eval_env = Monitor(
        RacingEnv(track=args.track),
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

    if args.resume_model:
        model = PPO.load(str(args.resume_model), env=train_env, device="cpu")
        model.set_random_seed(args.seed)
        model.tensorboard_log = str(run_dir / "tensorboard")
    else:
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
        reset_num_timesteps=args.resume_model is None,
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
