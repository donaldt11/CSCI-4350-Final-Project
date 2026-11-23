from agent import QLearningAgent


def main():
    agent = QLearningAgent(epsilon=0.0)

    state = (0, 0, 0)
    next_state = (0, 1, 0)
    action = 3  # right

    chosen_action = agent.choose_action(state)
    print("Chosen action:", chosen_action)

    agent.learn(
        state=state,
        action=action,
        reward=10,
        next_state=next_state,
        done=False,
    )

    learned_value = agent.q_table[0, 0, 0, action]

    assert learned_value > 0, "Q-value should increase after reward."
    print("Q-learning update test passed.")
    print("Updated Q-value:", learned_value)


if __name__ == "__main__":
    main()