# Racing Agent Practice Experiment

## Research question

How does additional training practice on a technical racing circuit affect agents that have first reached a common baseline on a simple oval?

## Design

Use a two stage, matched checkpoint design. In stage one, train independent PPO agents on the simple oval until each meets the baseline criterion. Save each qualifying checkpoint. In stage two, copy each qualifying checkpoint into every practice duration group and continue training the copies on the technical circuit. Include a zero practice group that receives evaluation only. This makes the baseline checkpoint the matched starting point for the practice comparisons.

Use at least five independent baseline seeds if compute time allows. Set the baseline criterion before starting. The recommended criterion is completion of at least 18 of 20 oval evaluation episodes. Set a maximum baseline training budget, such as 2,000,000 environment steps. If a seed does not meet the criterion by that limit, record it as a baseline failure and adjust the learning setup before the practice comparison.

Choose practice doses before running the study, for example 0, 0.5, 1, and 2 hours. Each nonzero dose is timed by `practice_agent.py`. The script records requested hours, actual elapsed training seconds, additional environment steps, seed, and the source checkpoint. Run all groups on the same computer with the same settings. Wall clock time is the assigned practice dose; environment steps are recorded because training speed varies.

## Environment

The project has two tracks with the same observation and action spaces. The baseline oval is a smooth ellipse with a 50 pixel half width. The technical circuit is an asymmetric multi-radius route defined by 72 control points and a 22 pixel half width. Its radial profile is r = 1 + 0.10 sin(3θ + 0.25) + 0.04 cos(5θ - 0.55) + 0.02 sin(7θ + 0.7), mapped to screen coordinates x = 500 + 340r cos(θ) and y = 350 + 235r sin(θ). Both routes are interpolated with periodic Catmull-Rom splines using eight samples per control point interval.

Each reset samples a lateral offset from -8 to 8 pixels, a heading adjustment from -5 to 5 degrees, and a starting speed from 0 to 0.4 pixels per step. The Gymnasium reset seed makes these start conditions reproducible. The agent receives a 10 value observation containing heading error, normalized lateral offset and speed, two route lookaheads, lap progress, and departure allowance. The discrete action space is `MultiDiscrete([3, 5])`: brake, coast, or accelerate combined with one of five steering levels.

An episode ends after one completed lap, six track departures, 100 consecutive off track steps, or 4,000 steps. Reward is route progress multiplied by 0.04, minus 0.01 per step. On track speed adds 0.01 times current speed. Each off track step costs 0.15, entering an off track event costs 3, finishing a lap adds 100, and reaching the stranded limit costs an additional 20.

## Training

Train each baseline agent with `train_agent.py --track oval`. The default PPO policy uses two 128 unit hidden layers, learning rate 0.0003, rollout length 1,024, minibatches of 128, 10 optimization epochs, discount factor 0.99, generalized advantage estimation factor 0.95, and entropy coefficient 0.01. The default run is 500,000 environment steps with CPU training. If an agent has not reached the baseline criterion, resume from its saved final model with `--resume-model`.

For practice, load the qualifying oval checkpoint and run `practice_agent.py` on the technical track for the assigned wall clock duration. The zero practice group keeps the baseline checkpoint unchanged. Keep the training seed and assigned practice duration in each run record.

## Evaluation and comparison

Before practice, evaluate each baseline checkpoint on the technical track using the same held out seed set that will be used after practice. Evaluate every resulting model on those same technical track seeds. The evaluator uses deterministic policy actions while reset seeds create reproducible start variation.

Primary outcomes are technical track completion rate and lap progress. Also report track departures, on track fraction, and steps per completed lap. For each matched baseline checkpoint, calculate the after practice value minus its before practice value. Summarize those changes within each practice duration group. Report actual practice seconds and additional environment steps alongside the assigned hours. Do not claim improvement until these evaluations are run.

## Commands

From the `Racing_sim` directory, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train an oval baseline:

```powershell
python train_agent.py --track oval --timesteps 500000 --seed 42
```

Check whether it meets the baseline criterion:

```powershell
python evaluate_agent.py --track oval --model training_runs/<baseline-run>/models/final_model.zip --episodes 20 --seed 1000 --output training_runs/oval_baseline_seed42.csv
```

If needed, continue baseline training from the saved model:

```powershell
python train_agent.py --track oval --timesteps 500000 --seed 42 --resume-model training_runs/<baseline-run>/models/final_model.zip
```

Measure baseline transfer to the technical track before practice:

```powershell
python evaluate_agent.py --track technical --model training_runs/<baseline-run>/models/final_model.zip --episodes 20 --seed 2026 --output training_runs/technical_before_seed42.csv
```

Continue one copy for a one hour practice condition:

```powershell
python practice_agent.py --baseline-model training_runs/<baseline-run>/models/final_model.zip --practice-hours 1 --seed 42 --label one_hour_seed42
```

Evaluate the final practice model with the same technical seeds, replacing the run path:

```powershell
python evaluate_agent.py --track technical --model training_runs/<practice-run>/final_model.zip --episodes 20 --seed 2026 --output training_runs/technical_after_seed42.csv
```

TensorBoard logs are stored in each run folder. Open them with `tensorboard --logdir training_runs`.

## Draft methods paragraph

Independent PPO agents were first trained on a smooth oval circuit until they completed at least 18 of 20 seeded evaluation episodes. Each qualifying model was saved as a baseline checkpoint. Copies of each checkpoint were assigned to practice duration conditions of [insert selected durations], during which the policy continued training on a technical circuit. A zero practice condition retained the baseline checkpoint without additional training. The technical circuit was generated from 72 control points sampled from a radial profile containing third, fifth, and seventh harmonics and interpolated with a periodic Catmull-Rom spline. Agents were evaluated on the technical circuit before and after practice using the same 20 seeded starting conditions and deterministic policy actions. Outcomes were completion rate, lap progress, departures, on track fraction, and steps per completed lap. Actual elapsed training time and additional environment steps were recorded for each practice run. [Insert replicate count, selected practice durations, training settings, and observed results.]
