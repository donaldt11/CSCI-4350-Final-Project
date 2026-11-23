import random

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        grid_size=6,
        num_actions=6,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.num_actions = num_actions

        # State robot row, robot column, carrying package
        # Action: up, down, left, right, pickup, drop
        self.q_table = np.zeros((grid_size, grid_size, 2, num_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        row, col, carrying = state
        return int(np.argmax(self.q_table[row, col, carrying]))

    def learn(self, state, action, reward, next_state, done):
        row, col, carrying = state
        next_row, next_col, next_carrying = next_state

        current_q = self.q_table[row, col, carrying, action]
        best_next_q = np.max(
            self.q_table[next_row, next_col, next_carrying]
        )

        target = reward if done else (
            reward + self.discount_factor * best_next_q
        )

        self.q_table[row, col, carrying, action] += (
            self.learning_rate * (target - current_q)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )