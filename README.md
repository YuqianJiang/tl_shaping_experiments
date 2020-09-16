# Experiments for Temporal-Logic-Based Reward Shaping for Continuing Learning Tasks

This repository contains the source code of the experiments for the paper Temporal-Logic-Based Reward Shaping for Continuing Learning Tasks.

## Requirements

To automatically install all dependencies:

```setup
pip install -r requirements.txt
```

Our deep R-learning algorithm is implemented on top of [OpenAI Baselines](https://github.com/openai/baselines). Follow their instructions in case manual installation is required.

## Cart Pole

### Environment

The environment `cartpole_continuing.py` is modified from the standard cart pole environment in [OpenAI Gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py). The continuing cart pole environment removes the episode termination conditions and allows the cart and the pole to be at any positions.

### Training

To train the cart pole agent with our temporal-logic-based shaping method, run this command:

```train
python continuing_cartpole/train_cartpole.py
```

## Continual Area Sweeping

### Environment

The environment `env.py` implements a robot sweeping repeatedly and non-uniformly to maximize average reward in a grid world. Rewards/events appear in grid cells with different probabilities, and the robot receives a reward by going to each cell with an active event (such as trash to be picked up). There are two different scenarios studied in the paper: 1) events only appear in a certain region, 2) a human moves around and may generate event with every step.

### Training

To run the experiment in the "always kitchen" scenario:

```train
python continual_area_sweeping/shield_experiment_region.py
```

To run the experiment in the "always keep human visible" scenario:

```train
python continual_area_sweeping/shield_experiment_person.py
```

To run the experiment in the "always keep human visible and always corridor" scenario:

```train
python continual_area_sweeping/experiment_conjunction.py
```

## Grid World

### Environment

`gridworld.py` implements a continuing grid world environment where the agent receives a reward and gets "transported" to a random cell when it reaches some "goal" cell. In this experiment, the "goal" cell is on the bottom right.

### Training

Shaping and shielding methods with standard R-learning are implemented. To train the agent, run this command:

```train
python gridworld/continuing_experiment.py --csv results.csv
```
