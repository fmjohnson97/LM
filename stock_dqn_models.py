import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2

# picks an action as a number between 0 and limit
def sampleAction(limit):
    return random.randint(0, limit)  # [0:sell, 1:hold, 2:buy]

class Replay(object):
    # memory for training the DQN
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
        self.fc = nn.Linear(sectors*4, 2)
        self.device = torch.device('cpu')
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

class Manager(object):
    # inits variables needed for DQN learning
    def __init__(self, sectors):
        self.sectors = sectors
        self.discount_factor = .99
        self.last_value= 0.0
        self.decay_factor = np.prod(sectors, dtype=float) / 10.0
        self.device=torch.device('cpu')
        # num sectors=#rows; num actions=#columns
        self.dqn = DQN(3).to(self.device)
        self.target = DQN(3).to(self.device)
        self.dqn_opt = optim.Adam(self.dqn.parameters())
        self.dqn_opt.zero_grad()
        self.loss = nn.MSELoss()

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def select_action(self, state, explore_rate):  # epsilon greedy strategy
        action=[]
        # Select a random action
        if random.random() <= explore_rate:
            for _ in state:
                action.append(sampleAction(1))
            action=torch.FloatTensor(action)
        # Select the action with the highest q
        else:
            with torch.no_grad():
                state=torch.FloatTensor(state).view(1,len(state),3).to(self.device)
                action1 = self.dqn.forward(state)  # .view(self.sectors)
                action = torch.argmax(action1, 2).view(self.sectors)
            # [ 0:hold, 1: act]
        return action

    def updateDQN(self, replay):
        s0, s, a, r = replay.sample()
        s0 = torch.FloatTensor(s0).view(1, len(s0), 3).to(self.device)
        s = torch.FloatTensor(s).view(1, len(s), 3).to(self.device)
        dqn_output = self.dqn.forward(s0)
        tempr = np.zeros((self.sectors, 2), dtype=float)
        for i, act in enumerate(a.tolist()):
            if act != -1:
                tempr[i, int(act)] = r
        target_output = self.target.forward(s) * self.discount_factor + torch.FloatTensor(tempr)
        l = self.loss(dqn_output, target_output)
        self.dqn_opt.zero_grad()
        l.backward()
        self.dqn_opt.step()
        return l.item()

    def updateTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())


class Worker(object):
    # inits variables needed for DQN learning
    def __init__(self, sectors):
        self.sectors = sectors
        self.discount_factor = .99
        self.goal=None
        self.device=torch.device('cpu')
        self.decay_factor = np.prod(sectors, dtype=float) / 10.0
        # num sectors=#rows; num actions=#columns
        self.dqn = DQN(3).to(self.device)
        self.target = DQN(3).to(self.device)
        self.dqn_opt = optim.Adam(self.dqn.parameters())
        self.dqn_opt.zero_grad()
        self.loss = nn.MSELoss()

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def select_action(self, state, explore_rate):  # epsilon greedy strategy
        action = []
        # Select a random action
        if random.random() <= explore_rate:
            for _ in state:
                action.append(sampleAction(1))
            for i,g in enumerate(self.goal):
                if g==0:
                    action[i]=-1
            action = torch.FloatTensor(action)
        # Select the action with the highest q
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).view(1, len(state),3).to(self.device)
                action1 = self.dqn.forward(state)  # .view(self.sectors)
                action = torch.argmax(action1, 2).view(self.sectors)
            for i,g in enumerate(self.goal):
                if g==0:
                    action[i]=-1
            # [ 0:sell, 1: buy]
        return action

    def updateDQN(self,replay):
        s0, s, a, r = replay.sample()
        s0 = torch.FloatTensor(s0).view(1, len(s0), 3).to(self.device)
        s = torch.FloatTensor(s).view(1, len(s), 3).to(self.device)
        dqn_output = self.dqn.forward(s0)
        tempr = np.zeros((self.sectors, 2), dtype=float)
        for i, act in enumerate(a.tolist()):
            if act!=-1:
                tempr[i,int(act)]=r[i]
        target_output = self.target.forward(s) * self.discount_factor + torch.FloatTensor(tempr)
        l = self.loss(dqn_output, target_output)
        self.dqn_opt.zero_grad()
        l.backward()
        self.dqn_opt.step()
        return l.item()

    def updateTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())