import numpy as np
import random
import math
import torch.nn as nn
import torch
import torch.optim as optim

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2


# picks an action as a number between 0 and 3
def sampleAction():
    return random.randint(0, 3)  # [0:"N", 1:"S", 2:"E", 3:"W"]

class Replay(object):
    def __init__(self):
        self.max_size = 50
        self.transitions = []  # (old state, new state, action, reward)

    def store(self, s0, s, a, r):
        self.transitions.append((s0, s, a, r))
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def sample(self):
        return self.transitions[random.randint(0, len(self.transitions) - 1)]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.LSTM(2, 4, num_layers=32)
        self.layer2 = nn.LSTM(4, 8, num_layers=64)
        self.layer3 = nn.LSTM(8, 16, num_layers=64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, h = self.layer1(x)
        x=self.relu(x)
        x,h=self.layer2(x)
        x=self.relu(x)
        x,h=self.layer3(x)
        x = self.relu(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x


class Manager(object):
    # inits variables needed for Q learning
    def __init__(self, maze_size, discount_factor=0.99):
        self.decay_factor = np.prod(maze_size, dtype=float) / 10.0
        self.discount_factor = discount_factor
        self.maze_ind = 0
        self.maze_size = maze_size
        self.device=torch.device('cuda')
        self.dqn=DQN().to(self.device)
        self.target=DQN().to(self.device)
        self.dqn_opt=optim.Adam(self.dqn.parameters())
        self.loss=nn.MSELoss()

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    # executes the epsilon greed exploration stategy
    def select_action(self, state, er):
        # Select a random action
        if random.random() < er:
            action = sampleAction()
        # Select the action with the highest q
        else:
            state = torch.FloatTensor([state[0], state[1]]).view(1, 1, len(state)).to(self.device)
            val, action = self.dqn.forward(state)[0][0].max(0)
            action = action.item()
        return action

    # updates the DQN
    def updateDQN(self, replay, s0=None, s=None, r=None):
        if replay is not None:
            s0, s, a, r = replay.sample()
        s0=torch.FloatTensor(s0).view(1,1,len(s0)).to(self.device)
        s=torch.FloatTensor(s).view(1,1,len(s)).to(self.device)
        dqn_output=self.dqn.forward(s0)
        target_output=self.target.forward(s)*self.discount_factor+r
        l=self.loss(dqn_output, target_output)
        self.dqn_opt.zero_grad()
        l.backward()
        self.dqn_opt.step()
        return l.item()

    #update the target network
    def updateTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())


class Worker(object):
    # inits variables needed for Q learning
    def __init__(self, maze_size, goal=None, discount_factor=0.99):
        self.goal = goal
        self.decay_factor = np.prod(maze_size, dtype=float) / 10.0
        self.discount_factor = discount_factor
        self.max_t = np.prod(maze_size, dtype=int) * 100
        self.maze_ind = 1
        self.maze_size = maze_size
        self.device = torch.device('cuda')
        self.dqn=DQN().to(self.device)
        self.target=DQN().to(self.device)
        self.dqn_opt = optim.Adam(self.dqn.parameters())
        self.loss = nn.MSELoss()

    # computes the exploration and learning rates since they change based on time step
    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    # executes the epsilon greed exploration stategy
    def select_action(self, state, er):
        # Select a random action
        if random.random() < er:
            action = sampleAction()
        # Select the action with the highest q
        else:
            state = torch.FloatTensor([state[0], state[1]]).view(1, 1, len(state)).to(self.device)
            val, action = self.dqn.forward(state)[0][0].max(0)
            action = action.item()
        return action

    # updates the DQN
    def updateDQN(self, replay):
        s0, s, a, r = replay.sample()
        s0 = torch.FloatTensor(s0).view(1, 1, len(s0)).to(self.device)
        s = torch.FloatTensor(s).view(1, 1, len(s)).to(self.device)
        dqn_output = self.dqn.forward(s0)
        target_output = self.target.forward(s) * self.discount_factor + r
        l = self.loss(dqn_output, target_output)
        self.dqn_opt.zero_grad()
        l.backward()
        self.dqn_opt.step()
        return l.item()

    # update the target network
    def updateTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())
