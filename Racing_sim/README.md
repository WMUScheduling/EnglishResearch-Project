# Racing Simulation

This folder contains a keyboard driving demo and a PPO reinforcement learning experiment on a fixed, technical circuit.

## Install

Run these commands from this folder while your project virtual environment is active:

```powershell
python -m pip install -r requirements.txt
```

## Preview the circuit

```powershell
python main.py
```

Use the arrow keys or WASD to drive. Hold Shift while steering for a stronger turn.

## Train PPO

```powershell
python train_agent.py --timesteps 500000 --seed 42
```

Each run creates a timestamped folder under `training_runs`. It records the training configuration, Monitor CSV files, evaluation results, TensorBoard events, periodic checkpoints, the best evaluated model, and the final model.

View live training statistics in a second terminal:

```powershell
tensorboard --logdir training_runs
```

## Evaluate and watch a trained policy

Replace `<run-folder>` with the folder name printed after training:

```powershell
python evaluate_agent.py --model training_runs/<run-folder>/models/best/best_model.zip --episodes 20 --seed 2026 --output training_runs/evaluation_results.csv
python run_agent.py --model training_runs/<run-folder>/models/best/best_model.zip
```

The evaluator uses reproducible randomized start conditions and reports completion rate, lap progress, departures, and on-track fraction. The methods draft is in [EXPERIMENT_METHODS.md](EXPERIMENT_METHODS.md).

Models saved before the multi-turn environment was added use the previous observation and action spaces. Train a new model before evaluating or running the updated environment.
