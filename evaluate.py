import random

import numpy as np

from agent import QLearningAgent
from environment import WarehouseEnvironment


EVALUATION_EPISODES = 100


def evaluate_trained_agent(env, agent):
    successes = 0
    rewards = []
    successful_steps = []

    for _ in range(EVALUATION_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        success = False

        while not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward

            if reward == 100:
                success = True

        rewards.append(total_reward)

        if success:
            successes += 1
            successful_steps.append(env.steps)

    return successes, rewards, successful_steps


def evaluate_random_agent(env):
    random_generator = random.Random(42)

    successes = 0
    rewards = []

    for _ in range(EVALUATION_EPISODES):
        env.reset()
        done = False
        total_reward = 0

        while not done:
            action = random_generator.randint(0, 5)
            _, reward, done = env.step(action)
            total_reward += reward

            if reward == 100:
                successes += 1

        rewards.append(total_reward)

    return successes, rewards


def print_trained_results(successes, rewards, successful_steps):
    print("\nTrained Q-learning robot")
    print(
        f"Success rate: "
        f"{successes / EVALUATION_EPISODES * 100:.1f}%"
    )
    print(f"Average reward: {np.mean(rewards):.2f}")

    if successful_steps:
        print(
            "Average successful-delivery steps: "
            f"{np.mean(successful_steps):.2f}"
        )


def print_random_results(successes, rewards):
    print("\nRandom robot baseline")
    print(
        f"Success rate: "
        f"{successes / EVALUATION_EPISODES * 100:.1f}%"
    )
    print(f"Average reward: {np.mean(rewards):.2f}")


def main():
    env = WarehouseEnvironment()

    trained_agent = QLearningAgent(
        grid_size=env.grid_size,
        epsilon=0.0,
    )
    trained_agent.q_table = np.load("models/q_table.npy")

    trained_results = evaluate_trained_agent(env, trained_agent)
    random_results = evaluate_random_agent(env)

    print_trained_results(*trained_results)
    print_random_results(*random_results)


if __name__ == "__main__":
    main()