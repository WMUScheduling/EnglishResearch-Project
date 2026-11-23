# Racing Simulation

This project studies how reward and penalty magnitudes affect driving behavior in a PPO racing agent.

## Install

Run from this folder with the project virtual environment active:

```powershell
python -m pip install -r requirements.txt
```

## Preview a track

Open a static track preview before training. Close the window to exit.

```powershell
python preview_track.py --track complex
```

Available tracks are `oval`, `technical`, and `complex`. The complex course uses linked S turns with stronger changes in direction. Preview the course, then keep the selected track fixed throughout the pilot and experiment.

## Establish the pilot baseline

The pilot checks whether the agent can learn to drive the selected track under the default consequence values.

```powershell
python train_agent.py --track complex --timesteps 500000 --seed 42
python evaluate_agent.py --track complex --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 1000
```

Use the pilot to settle the track and training setup. Freeze them before the main experiment.

## Main experiment

Each condition starts a fresh independent PPO run. In the first stage, hold the completion bonus at 100 while varying progress reward and off-track penalty. Then hold a selected R/P setting constant and vary the completion bonus. For example:

```powershell
python train_agent.py --track complex --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --completion-bonus 100 --timesteps 500000 --seed 42
python evaluate_agent.py --track complex --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --completion-bonus 100 --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 2026 --output training_runs/<run-folder>/evaluation/final.csv
```

Repeat each condition across independent seeds with the same number of training steps. Do not resume from another condition's model in the main experiment.

## Training records

Each run stores the progress reward scale, off-track penalty scale, completion bonus, track, seed, PPO settings, Monitor episode CSV, checkpoints, evaluation results, the final model, and TensorBoard events under `training_runs`. The evaluator reports completion, progress, departures, speed, low speed time, and on track fraction. These measures help distinguish improved driving from simply slowing down or stopping.

View training data with:

```powershell
tensorboard --logdir training_runs
```

See [EXPERIMENT_METHODS.md](EXPERIMENT_METHODS.md) for the design and methods draft.
