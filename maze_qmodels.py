import numpy as np
import random
import math

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2


# picks an action as a number between 0 and 3
def sampleAction():
    return random.randint(0, 3)  # [0:"N", 1:"S", 2:"E", 3:"W"]


class Manager(object):
    # inits variables needed for Q learning
    def __init__(self, maze_size, num_actions, discount_factor=0.99):
        self.qtable = np.zeros(maze_size + (num_actions,), dtype=float)
        self.decay_factor = np.prod(maze_size, dtype=float) / 10.0
        self.discount_factor = discount_factor
        self.max_t = np.prod(maze_size, dtype=int) * 100
        self.solved_t = np.prod(maze_size, dtype=int)
        self.maze_ind = 0
        self.maze_size = maze_size

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    # executes the epsilon greed exploration stategy
    def select_action(self, state, er):
        if random.random() < er:
            action = sampleAction()  # chooses random action
        else:
            action = int(np.argmax(self.qtable[state]))  # chooses action with highest Q table value
        return action

    # updates the Q table
    def updateQ(self, state_0, state, action, reward, lr):
        best_q = np.amax(self.qtable[state])
        self.qtable[state_0 + (action,)] += lr * (
                reward + self.discount_factor * (best_q) - self.qtable[state_0 + (action,)])


class Worker(object):
    # inits variables needed for Q learning
    def __init__(self, maze_size, num_actions, goal=None, discount_factor=0.99):
        self.qtable = np.zeros(maze_size + (num_actions,), dtype=float)
        self.goal = goal
        self.decay_factor = np.prod(maze_size, dtype=float) / 10.0
        self.discount_factor = discount_factor
        self.max_t = np.prod(maze_size, dtype=int) * 100
        self.solved_t = np.prod(maze_size, dtype=int)
        self.maze_ind = 1
        self.maze_size = maze_size

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    # executes the epsilon greed exploration stategy
    def select_action(self, state, er):
        if random.random() < er:
            action = sampleAction()  # chooses random action
        else:
            action = int(np.argmax(self.qtable[state]))  # chooses action with highest Q table value
        return action

    # updates the Q table
    def updateQ(self, state_0, state, action, reward, lr):
        best_q = np.amax(self.qtable[state])
        self.qtable[state_0 + (action,)] += lr * (
                reward + self.discount_factor * (best_q) - self.qtable[state_0 + (action,)])
