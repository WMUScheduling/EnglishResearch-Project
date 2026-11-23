# Racing Agent Experiment Methods

## Research question

Can a Proximal Policy Optimization (PPO) agent learn to complete a technical, multi-turn racing circuit while maintaining track position?

## Environment

The task uses a custom Gymnasium environment implemented in `racing_env.py`. The circuit is a fixed closed route defined by 43 two-dimensional control points. A periodic Catmull-Rom spline interpolates the points, with eight samples per control-point interval. The resulting centerline is approximately 2,700 screen pixels long, and the paved track extends 42 pixels to either side of the centerline. The route contains linked bends, direction changes, a chicane-like section, and tighter turns. Vehicle position, speed, and heading update once per environment step. The initial position and heading are fixed at the start line. The episode ends after one completed lap, six track departures, 100 consecutive off-track steps, or 4,000 steps.

## Agent interface

At each step, the agent receives a 10-value observation: sine and cosine of heading error, normalized lateral offset, normalized speed, forward and lateral coordinates to centerline lookahead points at 55 and 125 pixels, lap progress, and the fraction of allowed departures used. The discrete action space is `MultiDiscrete([3, 5])`. Throttle actions brake, coast, or accelerate. Steering has five levels from strong left to strong right.

Reward is the change in route distance multiplied by 0.04, minus a per-step cost of 0.01. On-track speed adds 0.01 times current speed. Each off-track step costs 0.15, and entering an off-track event costs an additional 3. Completing a lap adds 100. Remaining off track for 100 steps costs an additional 20. A lap completion is recorded as success.

## Training procedure

The default trainer uses Stable-Baselines3 PPO with an MLP policy containing two 128-unit hidden layers. The default training budget is 500,000 environment steps with random seed 42. Hyperparameters are learning rate 0.0003, rollout length 1,024, minibatch size 128, 10 optimization epochs per update, discount factor 0.99, generalized advantage estimation factor 0.95, and entropy coefficient 0.01. Training runs on CPU.

The trainer evaluates a separate environment every 25,000 steps for five deterministic episodes and saves the best performing model. It saves model checkpoints every 50,000 steps. Each run directory contains the exact configuration in `config.json`, episode-level training and evaluation Monitor CSV files, Stable-Baselines3 evaluation arrays, checkpoints, the final model, and TensorBoard event data. TensorBoard records episode reward and PPO training statistics.

## Evaluation

After training, `evaluate_agent.py` evaluates the chosen model over 20 deterministic episodes with seeds 2026 through 2045. It writes one row per episode to a CSV with return, lap progress, completion, departures, stranded status, episode length, and on-track fraction. Report the mean and standard deviation of these measures across episodes. Do not report model performance until the evaluation script has been run.

## Reproduction commands

From the `Racing_sim` directory, install dependencies and start training:

```powershell
python -m pip install -r requirements.txt
python train_agent.py --timesteps 500000 --seed 42
```

Open the live training dashboard in another terminal:

```powershell
tensorboard --logdir training_runs
```

Evaluate the best model by substituting the run folder printed by the trainer:

```powershell
python evaluate_agent.py --model training_runs/<run-folder>/models/best/best_model.zip --episodes 20 --seed 2026 --output training_runs/evaluation_results.csv
```

Run the trained agent on screen:

```powershell
python run_agent.py --model training_runs/<run-folder>/models/best/best_model.zip
```

## Draft methods paragraph

A Proximal Policy Optimization agent was trained in a custom Gymnasium racing environment to complete a fixed, technical, multi-turn circuit. The circuit was defined by 43 two-dimensional control points and interpolated with a periodic Catmull-Rom spline using eight samples per interval. The agent received a 10-element observation containing heading error, lateral offset, speed, two route lookaheads, lap progress, and remaining departure allowance. At each step, it selected one of three throttle actions and one of five steering actions. Training used Stable-Baselines3 PPO with a two-layer, 128-unit multilayer perceptron, a learning rate of 0.0003, 1,024-step rollouts, minibatches of 128, 10 optimization epochs, a discount factor of 0.99, generalized advantage estimation of 0.95, and an entropy coefficient of 0.01. The model was trained for 500,000 environment steps with seed 42. A separate environment was evaluated deterministically every 25,000 steps, and the best checkpoint was retained. Final performance was assessed over 20 episodes using seeds 2026 through 2045. [Insert completion rate, mean lap progress, mean departures, and mean on-track fraction after evaluation.]
