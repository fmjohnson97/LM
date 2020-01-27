from multi_trans_stock_env import MultiTransStock
from stock_qmodels import Worker
from matplotlib import pyplot as plt
import math
import numpy as np
import random


MIN_EXPLORE_RATE = 0.05
MIN_LEARNING_RATE = 0.01
NUM_SECTORS = 6
DECAY_FACTOR = np.prod(NUM_SECTORS, dtype=float) / 5.0
NUM_EPISODES = 50
MAX_T = 5000
TERMINAL_VALUE = 1000

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

# picks an action as a number between 0 and limit
def sampleAction(limit):
    return random.randint(0, limit)  # [0:sell, 1:hold, 2:buy]

def compute_price_diff(state):
    diff = []
    for i, s in enumerate(state):
        if s > env.last_price[i]:
            diff.append(1)  # up
        else:
            diff.append(0)  # down
    return diff

class AgentManager(object):
    def __init__(self, sectors):
        self.sectors=sectors
        self.discount_factor=.99
        self.q_table=np.zeros((self.sectors,2,3), dtype=float)

    def select_action(self, diff, er):
        action=[] # will give the number agent to use
        for s in range(self.sectors):
            if random.random()<=er:
                action.append(sampleAction(2))
            else:
                action.append(int(np.argmax(self.q_table[s,diff[s]])))

        return action

    def update_Q(self, action, reward, lr, od, d):
        for i,a in enumerate(action):
            bestQ=np.amax(self.q_table[i,d])
            self.q_table[i,od,a]+=lr*(reward+self.discount_factor*bestQ-self.q_table[i,od,a])


### Begin the Simulation ###
env=MultiTransStock(NUM_SECTORS)

manager=AgentManager(NUM_SECTORS)
# define workers with different trading strategies
workers=[Worker(NUM_SECTORS,1),
         Worker(NUM_SECTORS,2),
         Worker(NUM_SECTORS,3)]


fig = plt.figure()
plt.suptitle('Feudal Q Learning Agent - Fixed Transaction Workers')
ax = fig.add_subplot(111)

for episode in range(NUM_EPISODES):
    # setting the initial learning rates
    lr = get_learning_rate(episode)
    # setting the initial exploration rates
    er = get_explore_rate(episode)
    manager_actions = []
    # resetting the manager/worker environments
    p0=env.reset()
    env.update_value(p0)
    m_reward = 0
    w_reward = 0
    print(lr, er)
    for t in range(MAX_T):
        #select an action
        d0 = compute_price_diff(p0)
        action_m = manager.select_action(d0, er)
        manager_actions.append(action_m)
        old_val=env.value[-1]
        # make the worker move multiple times per manager instr
        for _ in range(1):
            diff_w = compute_price_diff(p0)
            #select and action
            action_w=[0]*NUM_SECTORS
            for i,w in enumerate(workers):
                w.goal=[1 if x==i else 0 for x in action_m]
                action_w =np.array(action_w)+np.array(w.select_action(diff_w, er))*np.array(w.goal)
            #execute the action and get the new prices and state
            p, rw = env.step(action_m, action_w) #not using this rm
            w_reward+=sum(rw)

            #update the q table
            for i,w in enumerate(workers):
                act=np.array(w.goal)*np.array(action_w)
                rew=np.array(w.goal)*np.array(rw)
                w.update_Q(action_w, rw, lr, diff_w)

            # render the worker graph
            env.render(ax)
            p0=p

            if env.value[-1] >= TERMINAL_VALUE:
                break
        new_val=env.value[-1]
        rm=(new_val-old_val)/new_val
        # update the q table
        d = compute_price_diff(p)
        manager.update_Q(action_m, rm, lr, d0,d)
        m_reward+=rm
        d=d0

        if env.value[-1]>=TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break
        elif env.value[-1]<=0:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break
        elif t >= MAX_T-1:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break