from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window=100):
    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def main():
    rewards = np.load("data/rewards.npy")
    steps = np.load("data/steps.npy")
    successes = np.load("data/successes.npy")

    window = 100
    episodes = np.arange(window, len(rewards) + 1)

    Path("outputs").mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(episodes, moving_average(rewards, window))
    axes[0].set_title("Average Reward During Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    axes[1].plot(episodes, moving_average(steps, window))
    axes[1].set_title("Average Steps Per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    axes[2].plot(
        episodes,
        moving_average(successes, window) * 100,
    )
    axes[2].set_title("Delivery Success Rate")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("outputs/training_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()