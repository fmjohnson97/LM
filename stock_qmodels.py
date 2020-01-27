import numpy as np
import random


# picks an action as a number between 0 and limit
def sampleAction(limit):
    return random.randint(0, limit)  # [0:sell, 1:hold, 2:buy]


class Manager(object):
    # inits variables needed for Q learning
    def __init__(self, sectors):
        self.sectors = sectors
        self.discount_factor = .99
        # num sectors=#rows; num actions=#columns
        self.q_table = np.zeros((self.sectors,2,2), dtype=float)

    def select_action(self, diff, explore_rate):  # epsilon greedy strategy
        action=[]
        for s in range(self.sectors):
            # Select a random action
            if diff[s]== 0 or diff[s]==1:
                d=0 # price went down
            else:
                d=1 #price went up
            if random.random() <= explore_rate:
                action.append(sampleAction(1))
            # Select the action with the highest q
            else:
                action.append(int(np.argmax(self.q_table[s, d])))
                # [ 0:hold, 1: act]
        return action

    def update_Q(self, action, reward, lr,diff):
        for i,a in enumerate(action):
            if diff[i]== 0 or diff[i]==1:
                d=0 # price went down
            else:
                d=1 #price went up
            bestQ = np.amax(self.q_table[i, d])
            self.q_table[i,d,a] += lr * (reward + self.discount_factor * bestQ - self.q_table[i,d,a])


class Worker(object):
    # inits variables needed for Q learning
    def __init__(self, sectors, transactions=1):
        self.sectors = sectors
        self.discount_factor = .99
        self.goal=None
        self.transactions=transactions
        # num sectors=#rows; num actions=#columns #
        self.q_table = np.zeros((self.sectors,4, 2), dtype=float)

    def select_action(self, diff, explore_rate):  # epsilon greedy strategy
        action=[]
        for i,g in enumerate(self.goal):
            # Select a random action
            if g==0:
                action.append(-1) #hold
            elif random.random() < explore_rate:
                action.append(sampleAction(1)) #pick between 0: sell or 1: buy
            # Select the action with the highest q
            else:
                action.append(int(np.argmax(self.q_table[i,diff[i]])))
                # [ -1:hold, 0: sell, 1:buy]
        return action

    def update_Q(self, actions, reward, lr, diff):
        for i, a in enumerate(actions):
            if a != -1:
                bestQ = np.amax(self.q_table[i, diff[i]])
                self.q_table[i, diff[i], a] += lr * (reward[i] + self.discount_factor * bestQ - self.q_table[i,diff[i], a])