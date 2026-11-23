# Reward and Penalty Magnitude Experiment

## Research question

How do changes in the magnitude of progress rewards and off-track penalties affect a PPO agent's driving behavior on a fixed simulated race track?

## Study design

First run a pilot using one fixed track and the default consequence values. The pilot is for establishing that the environment and PPO setup can produce a competent baseline driver. A recommended criterion is at least 18 completed laps in 20 seeded evaluation episodes. If the pilot does not meet that criterion, adjust the environment or training setup before collecting the main experiment. Once selected, freeze the track, observation and action spaces, reward components, PPO settings, and training budget.

For the main experiment, train agents independently from fresh initialization under each consequence setting. Do not fine-tune a shared baseline checkpoint. Keep the track, PPO algorithm, hyperparameters, number of environment steps, and evaluation seeds constant. Use multiple random seeds in every setting. A 3 by 3 factorial design is a practical starting point: progress reward scales of 0.02, 0.04, and 0.08 crossed with off-track penalty scales of 0.5, 1.0, and 2.0. These are starting levels for the pilot; finalize the levels before the main runs.

The progress reward scale multiplies route distance change. The off-track penalty scale multiplies the existing per-step off-track cost, departure cost, and stranded cost. The time cost, on-track speed bonus, lap completion reward, and training algorithm remain fixed. The run configuration records both scales for every agent.

## Environment and behavior measures

The simulator offers an oval, a technical multi-radius course, and a complex circuit. Preview the choices before the pilot, select one course, and use that same track for every main experiment condition. The oval uses 36 control points and is 50 pixels wide on either side of its centerline. The technical course uses 72 control points and is 22 pixels wide on either side. The complex circuit is generated from 72 radial control points with multiple linked bends and an 18 pixel half width. Its centerline is a single closed loop without track crossings. The agent's observation has 10 values: heading error, normalized lateral offset and speed, two route lookaheads, lap progress, and departure allowance. Actions combine one of three throttle choices with one of five steering choices.

The main dependent measures are lap progress, track departures, lap completion, and steps per completed lap. Also record average speed, fraction of time below the low-speed threshold, and on-track fraction. These measures help distinguish safer turn behavior from a policy that simply drives slowly or stops. Compare these behavior measures across conditions. Raw cumulative reward is not the primary outcome because its scale changes by design.

Episodes end after one completed lap, six track departures, 100 consecutive off-track steps, or 4,000 steps. Each step adds route progress multiplied by the progress reward scale, subtracts a fixed 0.01 time cost, and adds a fixed on-track speed bonus. Off-track step, departure, and stranded costs are multiplied by the off-track penalty scale. Lap completion adds a fixed 100 points.

## Training and recording

The default algorithm is Stable-Baselines3 PPO with an MLP policy containing two 128-unit hidden layers. Default settings are learning rate 0.0003, rollout length 1,024, minibatch size 128, 10 optimization epochs, discount factor 0.99, generalized advantage estimation factor 0.95, and entropy coefficient 0.01. The default budget is 500,000 environment steps on CPU. Each run stores its consequence values, track, seed, PPO settings, Monitor episode CSV, evaluation files, checkpoints, final model, and TensorBoard events under a unique run folder.

Use matched random seeds across consequence settings where possible, with at least five independent seeds per setting if compute time allows. For example, train all nine settings with seeds 42, 43, 44, 45, and 46. Evaluate policies with deterministic actions on the same held-out reset seeds. Report completion rate, lap progress, departures, average speed, low-speed fraction, on-track fraction, and successful lap time. Describe each replicate and do not treat individual episodes from one model as independent agents.

## Commands

From `Racing_sim`, install requirements:

```powershell
python -m pip install -r requirements.txt
```

Preview the complex course before training, then run the pilot on the selected fixed course:

```powershell
python train_agent.py --track oval --timesteps 500000 --seed 42
python evaluate_agent.py --track oval --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 1000
```

Run one main experiment condition:

```powershell
python train_agent.py --track oval --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --timesteps 500000 --seed 42
python evaluate_agent.py --track oval --progress-reward-scale 0.02 --off-track-penalty-scale 0.5 --model training_runs/<run-folder>/models/final_model.zip --episodes 20 --seed 2026 --output training_runs/<run-folder>/evaluation/final.csv
```

Run TensorBoard from a second terminal:

```powershell
tensorboard --logdir training_runs
```

## Draft methods paragraph

PPO agents were trained independently from fresh initialization in a custom Gymnasium racing environment. The study varied the magnitude of the reward for route progress and the penalty for leaving the track while holding the track, observation and action spaces, training algorithm, hyperparameters, and training budget constant. The progress reward scale was [insert levels], and the off-track penalty scale was [insert levels]. The penalty scale multiplied the per-step off-track cost, departure cost, and stranded cost. The time cost, on-track speed bonus, and lap completion reward remained fixed. Each agent was trained for [insert number] environment steps with [insert number] independent random seeds per condition. Performance was evaluated using deterministic actions on [insert number] common seeded starting conditions. Outcomes included lap progress, track departures, lap completion, steps per completed lap, average speed, low-speed fraction, and on-track fraction. [Insert observed results after training and evaluation.]
