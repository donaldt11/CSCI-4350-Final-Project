# Warehouse Package Delivery Robot

A reinforcement-learning project for CSCI-4350. A robot learns to navigate a
6x6 warehouse, avoid obstacles, pick up a package, and deliver it to a target
location.

The current version uses **tabular Q-learning**. Rather than receiving a
predefined route, the robot explores the warehouse and improves its policy
from rewards and penalties.

## Features

- 6x6 warehouse grid with walls and shelf obstacles
- Six actions: move up, down, left, right, pick up, and drop
- Rewards for successful pickup and delivery
- Penalties for invalid moves and invalid pickup/drop attempts
- Q-learning agent with epsilon-greedy exploration
- Training script that saves a learned Q-table

## Project files

| File | Purpose |
| --- | --- |
| `environment.py` | Defines the warehouse, robot state, actions, rewards, and rules. |
| `agent.py` | Defines the tabular Q-learning agent. |
| `train.py` | Trains the agent for multiple delivery episodes. |
| `test_environment.py` | Performs a basic environment test. |
| `test_agent.py` | Verifies that the agent updates a Q-value after feedback. |

## Setup

This project requires Python 3 and NumPy.

```powershell
python -m pip install numpy
```

## Run the tests

Test the warehouse environment:

```powershell
python .\test_environment.py
```

Test the Q-learning update:

```powershell
python .\test_agent.py
```

## Train the robot

Run the training loop:

```powershell
python .\train.py
```

The script trains the robot for 5,000 episodes and saves the learned Q-table
to `models/q_table.npy`.

## How the agent learns

The state is represented as:

```text
(robot row, robot column, carrying package)
```

For each state, the Q-table stores a value for each possible action. The robot
initially explores many random actions. After receiving a reward, it updates
the value for the action it took. Over time, actions that lead to successful
deliveries receive higher Q-values and are selected more often.

## Future improvements

- Compare the trained robot with a random baseline.
- Plot training rewards, steps, and delivery success rate.
- Support multiple warehouse layouts and randomized package locations.
- Compare learned routes with Breadth-First Search (BFS) shortest paths.
- Add a Pygame visualization of the trained robot.
