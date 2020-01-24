import sys
import numpy as np
import math
import random
from maze_env import MazeEnv
import torch
import torch.nn as nn
import torch.optim as optim


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


class Replay(object):
    #memory for training the DQN
    def __init__(self):
        self.max_size = 100
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

    def test(self):
        pass

    def select_action(self, state, er, device):
        # Select a random action
        if random.random() < er:
            action = env.action_space.sample()
        # Select the action with the highest q
        else:
            with torch.no_grad():
                state = torch.FloatTensor([state[0], state[1]]).view(1, 1, len(state)).to(device)
                val, action = self.forward(state)[0][0].max(0)
                action = action.item()
        return action


def simulate():
    # Instantiating the learning related parameters
    discount_factor = 0.99
    update_target = 5

    num_streaks = 0

    # Render the maze
    env.render()
    dqn = DQN()
    device = torch.device('cpu')
    dqn = dqn.to(device)
    target = DQN()
    target = target.to(device)
    dqn_opt = optim.Adam(dqn.parameters())
    dqn_opt.zero_grad()
    replay = Replay()
    loss = nn.MSELoss()

    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv = env.reset()

        # the initial state
        s0 = state_to_bucket(obv)
        total_reward = 0
        total_loss = 0

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

        for t in range(MAX_T):
            # Select an action
            action = dqn.select_action(s0, explore_rate, device)

            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            s = state_to_bucket(obv)
            total_reward += reward

            # store the transition
            replay.store(s0, s, action, reward)

            # update the dqn
            r_s0, r_s, r_a, r_r = replay.sample()
            r_s0 = torch.FloatTensor([r_s0[0], r_s0[1]]).view(1, 1, len(r_s0)).to(device)
            r_s = torch.FloatTensor([r_s[0], r_s[1]]).view(1, 1, len(r_s)).to(device)
            dqn_output =dqn.forward(r_s0)
            target_output=target.forward(r_s)
            dqn_opt.zero_grad()
            l = loss(dqn_output, target_output)
            l.backward()
            dqn_opt.step()
            total_loss+=l.item()

            # update the target network
            if episode % update_target == 0:
                target.load_state_dict(dqn.state_dict())

            # Setting up for the next iteration
            s0 = s

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f, loss = %f (streak %d)."
                      % (episode, t, total_reward, total_loss, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break


# Initialize the "maze" environment
maze_path= None #ToDo: replace this with the path to a maze file
maze_size = (4,4) #ToDo: instead of specifying a maze path, you can specify a maze size (x,y) and a random maze will be generated
gx=None #ToDo: None makes the goal in a random place. Change this to the x coordinate of your desired goal in the maze to have it fixed
gy=None #ToDo: None makes the goal in a random place. Change this to the y coordinate of your desired goal in the maze to have it fixed
env = MazeEnv(maze_file=maze_path, maze_size=maze_size, gx=gx, gy=gy )

'''
Defining the environment related constants
'''
# Number of discrete states (bucket) per state dimension
MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

'''
Learning related constants
'''
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

'''
Defining the simulation related constants
'''
NUM_EPISODES = 50000
MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
STREAK_TO_END = 100
SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
DEBUG_MODE = 0
RENDER_MAZE = True
ENABLE_RECORDING = True

'''
Creating a Q-Table for each state-action pair
'''
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

'''
Begin simulation
'''
simulate()
