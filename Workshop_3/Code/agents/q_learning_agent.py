import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((2**state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_size = state_size
        self.action_size = action_size

    def get_action(self, state):
        idx = self.state_to_index(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[idx])

    def update(self, state, action, reward, next_state):
        idx = self.state_to_index(state)
        next_idx = self.state_to_index(next_state)
        best_next = np.max(self.q_table[next_idx])
        self.q_table[idx][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[idx][action])

    def state_to_index(self, state):
        # Very simple binarization-based indexing (works if state is scaled [0,1])
        binary_state = (state > 0.5).astype(int)
        return int("".join(str(b) for b in binary_state), 2)
