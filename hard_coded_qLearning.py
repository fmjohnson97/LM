import numpy as np
from stock_env import StockEnv
from matplotlib import pyplot as plt

NUM_SECTORS = 6
NUM_EPISODES = 50
MAX_T = 5000
DECAY_FACTOR = np.prod(NUM_SECTORS, dtype=float) / 10.0
TERMINAL_VALUE = 1000
TRANS_FEE = 0.00


class q_agent(object):
    def __init__(self, sectors, bal=500):
        self.balance = bal
        self.sectors = sectors
        self.portfolio = [0] * sectors
        self.last_price = [1.0] * sectors
        self.value = []
        self.s_thresh = -.5
        self.b_thresh = .5

    def reset(self, bal=500):
        self.balance = bal
        self.portfolio = [0] * self.sectors
        self.last_price = [1.0] * self.sectors
        self.value = []

    def select_action(self, state): 
        diff = np.array(state) - np.array(self.last_price)
        action = []
        for d in diff:
            if d < self.s_thresh:
                action.append(-1)
            elif d > self.b_thresh:
                action.append(1)
            else:
                action.append(0)
        self.last_price = state
        return action

    def act(self, i, a, p):
        if a == -1 and self.portfolio[i] - 1 >= 0:  # sell
            self.balance += p - TRANS_FEE
            self.portfolio[i] -= 1
        elif a == 1 and self.balance - p - TRANS_FEE >= 0:  # buy
            self.balance -= (p + TRANS_FEE)
            self.portfolio[i] += 1
        # else is hold so you do nothing

    def update_value(self, prices):
        total = 0
        for i, p in enumerate(prices):
            total += self.portfolio[i] * p
        self.value.append(total + self.balance)


### Begin Simulation ###

env = StockEnv(NUM_SECTORS)
agent = q_agent(len(env.sectors))
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Hard Coded Agent')

for episode in range(NUM_EPISODES):
    # reset the environment and initialize the portfolio value
    agent.reset()
    p0 = env.reset()
    agent.update_value(p0)

    for t in range(MAX_T):
        # select the next action
        action = agent.select_action(p0)
        # execute the next action and get next state and reward
        p = env.step()

        for i, a in enumerate(action):
            agent.act(i, a, p[i])

        agent.update_value(p)

        # render the portfolio value graph
        env.render(ax, agent.value)

        # prepare for next iteration
        p0 = p

        if agent.value[-1] >= TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total value = %f"
                  % (episode, t, agent.value[-1]))
            break
        elif agent.value[-1] <= 0:
            print("Episode %d terminated after %f time steps with total value = %f. No more assets."
                  % (episode, t, agent.value[-1]))
            break
        elif t >= MAX_T-1:
            print("Episode %d timed out after %f time steps with total value = %f"
                  % (episode, t, agent.value[-1]))
            break
