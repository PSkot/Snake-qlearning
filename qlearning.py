import numpy as np
import random

class QLearning:
    def __init__(self, q_shape, alpha = 0.2, gamma = 0.7, epsilon = 0.3, epochs = 10000):
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploitation rate
        self.epochs = epochs
        self.Q = np.zeros(q_shape)

    def maxQ(self, state):
        return np.max(self.Q[state])

    def step(self, state, table_only = False):
        if np.random.random() > self.epsilon and not table_only:
            action = random.choice(range(self.Q.shape[1])) # Explore
        else:
            action = np.argmax(self.Q[state]) # Exploit
        return action

    def update(self, state, action, reward):
        Q_old = self.Q[state][action]
        self.Q[state][action] = Q_old + self.alpha * (reward + self.gamma*self.maxQ(action) - Q_old)
