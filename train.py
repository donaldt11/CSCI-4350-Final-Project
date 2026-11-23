from pathlib import Path

import numpy as np

from agent import QLearningAgent
from environment import WarehouseEnvironment


EPISODES = 5000


def main():
    env = WarehouseEnvironment()
    agent = QLearningAgent(grid_size=env.grid_size)

    rewards_per_episode = []
    successes = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        success = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            total_reward += reward

            if reward == 100:
                success = True

        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        successes.append(int(success))

        if (episode + 1) % 500 == 0:
            recent_success_rate = np.mean(successes[-500:]) * 100

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Reward: {total_reward:6.1f} | "
                f"Success rate: {recent_success_rate:5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    Path("models").mkdir(exist_ok=True)
    np.save("models/q_table.npy", agent.q_table)

    print("\nTraining complete.")
    print("Saved Q-table to models/q_table.npy")


if __name__ == "__main__":
    main()