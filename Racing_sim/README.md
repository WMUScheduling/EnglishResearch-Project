# Racing Simulation

This project compares reinforcement learning agents after baseline training and additional practice.

## Install and preview

Run from this folder with the project virtual environment active:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Use arrow keys or WASD to drive. Hold Shift while steering for a stronger turn.

## Stage one: train an oval baseline

```powershell
python train_agent.py --track oval --timesteps 500000 --seed 42
```

Evaluate it on 20 seeded oval episodes. Use the methods protocol to decide whether it meets the baseline criterion:

```powershell
python evaluate_agent.py --track oval --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 1000
```

If it has not met the criterion, continue training from its final model with `--resume-model`.

## Stage two: assign practice

Evaluate each baseline on the technical track before practice. Then continue a copy for the assigned practice duration:

```powershell
python evaluate_agent.py --track technical --model training_runs/<baseline-run>/models/final_model.zip --episodes 20 --seed 2026 --output training_runs/before_seed42.csv
python practice_agent.py --baseline-model training_runs/<baseline-run>/models/final_model.zip --practice-hours 1 --seed 42 --label one_hour_seed42
```

Evaluate the resulting `final_model.zip` on the same technical evaluation seeds used before practice. Repeat each practice condition across independent baseline seeds. The practice run records its requested time, actual elapsed time, and additional environment steps.

## Training records

Each run saves configuration, episode Monitor CSV files, model checkpoints, TensorBoard events, and final model weights under `training_runs`. View live training data with:

```powershell
tensorboard --logdir training_runs
```

See [EXPERIMENT_METHODS.md](EXPERIMENT_METHODS.md) for the baseline threshold, study design, full configuration, and methods draft.
