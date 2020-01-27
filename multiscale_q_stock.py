from multi_stock_env import MultiStock
from stock_qmodels import Worker, Manager
from matplotlib import pyplot as plt
import math
import numpy as np

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

def compute_price_diff(state):
    diff = []
    for i, s in enumerate(state):
        if s > env.last_price[i]:
            if env.portfolio[i] > 0:
                diff.append(3)  # up and have some
            else:
                diff.append(2)  # up and have none
        else:
            if env.portfolio[i] > 0:
                diff.append(1)  # down and have some
            else:
                diff.append(0)  # down and have none
    return diff


### Begin the Simulation ###
env=MultiStock(NUM_SECTORS)

manager=Manager(NUM_SECTORS)
worker = Worker(NUM_SECTORS)

fig = plt.figure()
plt.suptitle('Feudal Q Learning Agent - Transaction Fee')
ax = fig.add_subplot(111)
#ax2=fig.add_subplot(212) #ToDo: uncomment this and change the line above to get a graph of the agent's performance and the stock behavior

for episode in range(NUM_EPISODES):
    # setting the initial learning rates
    lr = get_learning_rate(episode)
    # setting the initial exploration rates
    er = get_explore_rate(episode)

    # resetting the manager/worker environments
    p0=env.reset()
    env.update_value(p0)
    m_reward = 0
    w_reward = 0
    print(lr, er)
    for t in range(MAX_T):
        #select an action
        diff_m = compute_price_diff(p0)
        action_m = manager.select_action(diff_m, er)

        # pass goal (sector) to the worker
        worker.goal= action_m
        old_val=env.value[-1]
        # make the worker move multiple times per manager instr
        for _ in range(8):
            diff_w = compute_price_diff(p0)
            #select and action
            action_w = worker.select_action(diff_w, er)
            #execute the action and get the new prices and state
            p, rw = env.step(action_m, action_w) #not using this rm
            w_reward+=sum(rw)

            #update the q table
            worker.update_Q(action_w, rw, lr, diff_w)

            # render the worker graph
            env.render(ax)
            # env.render(ax2) #ToDo: uncomment this to get a graph of the agent's performance and the stock behavior
            p0=p

            if env.value[-1] >= TERMINAL_VALUE:
                break
        new_val=env.value[-1]
        rm=(new_val-old_val)/new_val

        # update the q table
        manager.update_Q(action_m, rm, lr, diff_m)
        m_reward+=rm
        if env.value[-1]>=TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break
        elif env.value[-1]<=0:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break
        elif t*8 >= MAX_T-1:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, env.value[-1]))
            break