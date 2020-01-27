from stock_env import StockEnv
import numpy as np
import math
import random
from matplotlib import pyplot as plt

MIN_EXPLORE_RATE = 0.05
MIN_LEARNING_RATE = 0.01
NUM_SECTORS = 6
NUM_EPISODES = 50
MAX_T = 5000
DECAY_FACTOR = np.prod(NUM_SECTORS, dtype=float) / 5.0
TERMINAL_VALUE = 1000
TRANS_FEE = 0.10


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


class q_agent(object):
    def __init__(self, sectors, bal=500):
        self.balance = bal
        self.sectors = sectors
        self.portfolio = [0] * sectors
        self.last_price = [1.0] * sectors
        self.discount_factor = .99
        self.value = [1, 1]
        # num sectors=#rows; num actions=#columns
        self.q_table = np.zeros((self.sectors, 4, env.action_space.n), dtype=float)

    def reset(self, bal=500):
        self.balance = bal
        self.portfolio = [0] * self.sectors
        self.last_price = [1.0] * self.sectors
        self.value = [1, 1]

    def select_action(self, state, explore_rate):  # epsilon greedy strategy
        # Select a random action
        diff = []
        for i, s in enumerate(state):
            if s > self.last_price[i]:
                if self.portfolio[i] > 0:
                    diff.append(3)  # up and have some
                else:
                    diff.append(2)  # up and have none
            else:
                if self.portfolio[i] > 0:
                    diff.append(1)  # down and have some
                else:
                    diff.append(0)  # down and have none

        action = []
        if random.random() < explore_rate:
            for _ in state:
                action.append(env.action_space.sample())
        # Select the action with the highest q
        else:
            for i, d in enumerate(diff):
                action.append(int(np.argmax(self.q_table[i][d])))
            # [ 0:sell, 1: hold, 2:buy]
        return action

    def act(self, i, a, p):
        old_val = self.portfolio[i] * p
        if a == 0 and self.portfolio[i] - 1 >= 0:  # sell
            self.balance += (p - TRANS_FEE)
            self.portfolio[i] -= 1
            self.last_price[i] = p
            new_val = self.portfolio[i] * p
            r = (old_val - new_val) / old_val
        elif a == 2 and self.balance - (p + TRANS_FEE) >= 0:  # buy
            self.balance -= (p + TRANS_FEE)
            self.portfolio[i] += 1
            self.last_price[i] = p
            new_val = self.portfolio[i] * p
            r = (new_val - old_val) / new_val
        else:
            r = (p - self.last_price[i]) / p
        # else is hold so you do nothing
        # compute the reward if should end
        if (self.balance - (p + TRANS_FEE) < 0 and a == 2) or (self.portfolio[i] - 1 < 0 and a == 0):
            r = -1

        return r

    def update_Q(self, state, action, reward, lr):
        diff = []
        for i, s in enumerate(state):
            if s > self.last_price[i]:
                if self.portfolio[i] > 0:
                    diff.append(3)  # up and have some
                else:
                    diff.append(2)  # up and have none
            else:
                if self.portfolio[i] > 0:
                    diff.append(1)  # down and have some
                else:
                    diff.append(0)  # down and have none

        for i, a in enumerate(action):
            bestQ = np.amax(self.q_table[i, diff[i]])
            self.q_table[i, diff[i], a] += lr * (reward[i] + self.discount_factor * bestQ - self.q_table[i, diff[i], a])

    def update_value(self, prices):
        total = 0
        for i, p in enumerate(prices):
            total += self.portfolio[i] * p
        self.value.append(total + self.balance)


### Begin Simulation ###

env = StockEnv(NUM_SECTORS)
agent = q_agent(NUM_SECTORS)
fig = plt.figure()
plt.suptitle('Q Learning Agent')
ax = fig.add_subplot(111)
# ax2=fig.add_subplot(212) #ToDo: uncomment this and change the line above to get a graph of the agent's performance and the stock behavior


for episode in range(NUM_EPISODES):
    # get episodically changed parameters
    lr = get_learning_rate(episode)
    er = get_explore_rate(episode)
    print(lr, er)

    # reset the environment and initialize the portfolio value
    agent.reset()
    p0 = env.reset()
    agent.update_value(p0)
    total_reward = 0

    for t in range(MAX_T):

        # select the next action
        action = agent.select_action(p0, er)
        # execute the next action and get next state and reward
        p = env.step()
        reward = []
        for i, a in enumerate(action):
            r = agent.act(i, a, p[i])
            reward.append(r)
        for i, r in enumerate(reward):
            if r == 0.0:
                reward[i] = (agent.value[-1] - agent.value[-2]) / agent.value[-1]

        # update the q table
        agent.update_Q(p0, action, reward, lr)

        # prepare for next iteration
        p0 = p
        agent.update_value(p)
        for i in reward:
            total_reward += i

        # render the portfolio value graph
        env.render(ax, agent.value[2:])
        # env.render(ax2) #ToDo: uncomment this to get a graph of the agent's performance and the stock behavior

        if agent.value[-1] >= TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = %f. Total Value = %f"
                  % (episode, t, total_reward, agent.value[-1]))
            break
        elif agent.value[-1] <= 20:
            print("Episode %d terminated after %f time steps with total reward = %f. Total Value = %f. No more assets."
                  % (episode, t, total_reward, agent.value[-1]))
            break
        elif t >= MAX_T - 1:
            print("Episode %d terminated after %f time steps with total reward = %f. Total Value = %f"
                  % (episode, t, total_reward, agent.value[-1]))
            break
