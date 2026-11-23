# Racing Simulation

This project studies how reward and penalty magnitudes affect driving behavior in a PPO racing agent.

## Install and preview

Run from this folder with the project virtual environment active:

```powershell
python -m pip install -r requirements.txt
python main.py
```

## Establish the pilot baseline

Run the default consequence values on the oval and check whether the agent can complete laps:

```powershell
python train_agent.py --track oval --timesteps 500000 --seed 42
python evaluate_agent.py --track oval --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 1000
```

Use the pilot to settle the track and training setup. Freeze them before the main experiment.

## Main experiment

Each condition starts a fresh independent PPO run. Change only the progress reward scale and off-track penalty scale. For example:

```powershell
python train_agent.py --track oval --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --timesteps 500000 --seed 42
python evaluate_agent.py --track oval --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 2026 --output training_runs/<run-folder>/evaluation/final.csv
```

Repeat each condition across independent seeds with the same number of training steps. Do not resume from another condition's model in the main experiment.

## Training records

Each run stores consequence values, track, seed, PPO settings, episode Monitor CSV, checkpoints, evaluation results, the final model, and TensorBoard events under `training_runs`. The evaluator reports completion, progress, departures, speed, low-speed time, and on-track fraction. These behavior measures help distinguish improved driving from simply slowing down or stopping.

View training data with:

```powershell
tensorboard --logdir training_runs
```

See [EXPERIMENT_METHODS.md](EXPERIMENT_METHODS.md) for the design and methods draft.
