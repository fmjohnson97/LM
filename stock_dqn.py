from stock_env_dqn import StockEnvDQN
import numpy as np
import math
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

MIN_EXPLORE_RATE = 0.1
MIN_LEARNING_RATE = 0.15
NUM_SECTORS = 6
NUM_EPISODES = 20
MAX_T = 5000
DECAY_FACTOR = np.prod(NUM_SECTORS, dtype=float) / 5.0
TERMINAL_VALUE = 1000
TRANS_FEE = 2.00

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

class Replay(object):
    #memory for training the DQN
    def __init__(self):
        self.max_size = 1000
        self.transitions = []  # (old state, new state, action, reward)

    def store(self, s0, s, a, r):
        self.transitions.append((s0, s, a, r))
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def sample(self):
        return self.transitions[random.randint(0, len(self.transitions) - 1)]


class DQN(nn.Module):
    def __init__(self, sectors):
        super(DQN, self).__init__()
        self.layer1 = nn.LSTM(sectors, sectors*2, num_layers=32)
        self.layer2 = nn.LSTM(sectors*2, sectors*3, num_layers=64)
        self.layer3 = nn.LSTM(sectors*3, sectors*4, num_layers=64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(sectors*4, sectors)
        self.device = torch.device('cuda')
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.loss = nn.MSELoss()

    def forward(self, x):
        x, h = self.layer1(x)
        x=self.relu(x)
        x,h=self.layer2(x)
        x=self.relu(x)
        x,h=self.layer3(x)
        x = self.relu(x)
        x = self.fc(x)
        #x = self.tanh(x)
        return x


class q_agent(object):
    def __init__(self, sectors, bal=500):
        self.balance = bal
        self.sectors = sectors
        self.portfolio = [0] * sectors
        self.last_price = [1.0] * sectors
        self.discount_factor = .99
        self.value = []
        # num sectors=#rows; num actions=#columns
        self.dqn=DQN(3)
        self.target=DQN(3)
        self.dqn_opt=optim.Adam(self.dqn.parameters())
        self.dqn_opt.zero_grad()
        self.device=torch.device('cpu')
        self.loss=nn.MSELoss()

    def reset(self, bal=500):
        self.balance = bal
        self.portfolio = [0] * self.sectors
        self.last_price = [1.0] * self.sectors
        self.value = [1,1]

    def select_action(self, state, explore_rate):  # epsilon greedy strategy
        # Select a random action
        action = []
        if random.random() < explore_rate:
            for _ in state:
                action.append(env.action_space.sample())
            action=torch.FloatTensor(action)
        # Select the action with the highest q
        else:
            with torch.no_grad():
                state=torch.FloatTensor(state).view(1,len(state),3).to(self.device)
                action = self.dqn.forward(state)
                action= torch.argmax(action, 2).view(self.sectors)
            # [ -1:sell, 0: hold, 1:buy]
        return action

    def act(self, i, a, p):
        old_val = self.portfolio[i] * p
        if a == 0 and self.portfolio[i] - 1 >= 0:  # sell
            self.balance += (p - TRANS_FEE)
            self.portfolio[i] -= 1
            self.last_price[i] = p
        elif a == 2 and self.balance - p - TRANS_FEE >= 0:  # buy
            self.balance -= (p + TRANS_FEE)
            self.portfolio[i] += 1
            self.last_price[i] = p
        # else is hold so you do nothing
        new_val = self.portfolio[i] * p
        if new_val == 0:
            r = (self.value[-1]-self.value[-2])/self.value[-1]
        else:
            r = (new_val - old_val) / new_val
        # r = (self.value[-1] - self.value[-2]) / self.value[-1]
        # compute the reward if should end
        if (self.balance - p - TRANS_FEE < 0 and a == 2) or (self.portfolio[i] - 1 < 0 and a == 0):
            r = -1

        return r

    def update_Q(self, replay):
        s0, s, a, r = replay.sample()
        s0=torch.FloatTensor(s0).view(1,len(s0),3).to(self.device)
        s=torch.FloatTensor(s).view(1,len(s),3).to(self.device)
        dqn_output=self.dqn.forward(s0)
        tempr=[]
        for i in range(len(r)):
            tempr.append([r[i]]*3)
        target_output=self.target.forward(s)*self.discount_factor+torch.FloatTensor(tempr)
        l=self.loss(dqn_output,target_output)
        self.dqn_opt.zero_grad()
        l.backward()
        self.dqn_opt.step()
        return l.item()

    def updateTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())

    def update_value(self, prices):
        total = 0
        for i, p in enumerate(prices):
            total += self.portfolio[i] * p
        self.value.append(total + self.balance)


### Begin the Simulation ###
torch.manual_seed(30)
update_target=15
num_streaks = 0
env = StockEnvDQN(NUM_SECTORS)
agent=q_agent(NUM_SECTORS)
replay=Replay()
fig=plt.figure()
plt.suptitle('DQN Agent')
ax=fig.add_subplot(111)


for episode in range(NUM_EPISODES):
    # get episodically changed parameters
    lr = get_learning_rate(episode)
    er = get_explore_rate(episode)

    # reset the environment and initialize the portfolio value
    agent.reset()
    p0 = env.reset()
    agent.update_value(p0[-1])
    p0=np.array(p0).T.tolist()
    total_reward = 0
    total_loss=0

    for t in range(MAX_T):
        # select the next action
        action = agent.select_action(p0, er)
        # execute the next action and get next state and reward
        p = env.step()
        reward = []
        for i, a in enumerate(action):
            r = agent.act(i, a, p[-1][i])
            reward.append(r)
            total_reward+=r

        # save the transition
        agent.update_value(p[-1])
        p = np.array(p).T.tolist()
        if agent.value[-1] >= TERMINAL_VALUE:
            reward=[1]*NUM_SECTORS
        replay.store(p0, p, action, reward)

        # update the q table
        total_loss+=agent.update_Q(replay)

        # prepare for next iteration
        p0=p
        if episode%update_target==0:
            agent.updateTarget()

        # render the portfolio value graph
        env.render(ax, agent.value[2:])

        if agent.value[-1] >= TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = %f, loss = %f. Total Value = %f"
                  % (episode, t, total_reward, total_loss, agent.value[-1]))
            break
        elif agent.value[-1] <= 0:
            print("Episode %d terminated after %f time steps with total reward = %f, loss = %f. Total Value = %f. No more assets."
                  % (episode, t, total_reward, total_loss, agent.value[-1]))
            break
        elif t >= MAX_T-1:
            print("Episode %d terminated after %f time steps with total reward = %f, loss = %f. Total Value = %f"
                  % (episode, t, total_reward,total_loss, agent.value[-1]))
            break
