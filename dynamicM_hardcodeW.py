from multi_stock_env import MultiStock
from hardcoded_agents import AgentManager, ShortSellingAgent, LongSellingAgent, RandomAgent
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

def compute_price_diff(state, d):
    diff = []
    for i in range(NUM_SECTORS):
        if state[i]>env.last_price[i]:
            if d[i]==3 or d[i]==1:
                diff.append(3)  # up
            else:
                diff.append(1)
        else:
            if d[i]==3 or d[i]==1:
                diff.append(2)  # down
            else:
                diff.append(0)

    return diff

### Begin the Simulation ###
env=MultiStock(NUM_SECTORS)

manager=AgentManager(NUM_SECTORS)
workers=[RandomAgent(NUM_SECTORS),
         LongSellingAgent(NUM_SECTORS),
         ShortSellingAgent(NUM_SECTORS)]

fig=plt.figure()
plt.suptitle('Hard Coded Workers/Dynamic Manager - 1 per Stock')
ax=fig.add_subplot(111)

manager_actions=[]
worker_actions=[]

for episode in range(NUM_EPISODES):
    # setting the initial learning rates
    lr = get_learning_rate(episode)
    # setting the initial exploration rates
    er = get_explore_rate(episode)

    manager_actions = []
    worker_actions = []

    # resetting the manager/worker environments
    p0 = env.reset()
    env.update_value(p0)
    d0 = compute_price_diff(p0,[0]*NUM_SECTORS)
    m_reward = 0
    w_reward = [0]*NUM_SECTORS
    print(lr,er)

    for t in range(MAX_T):

        action_m = manager.select_action(d0,er)
        manager_actions.append(action_m)
        old_val = env.value[-1]
        old_rw=w_reward

        for _ in range(8):
            action_w=[]
            for i,a in enumerate(action_m):
                action_w.append(workers[a].select_action([p0[i], env.last_price[i], er]))
            worker_actions.append(action_w)

            p, rw = env.step(action_m, action_w)
            for i, r in enumerate(rw):
                w_reward[i]+=r
            env.render(ax)
            p0=p

            if env.value[-1] >= TERMINAL_VALUE:
                break

        rm=[]
        new_val=env.value[-1]
        new_rw= w_reward
        for i in range(NUM_SECTORS):
            if new_rw[i]==0:
                new_rw[i]=1
            rm.append((new_val-old_val)/new_val+(new_rw[i]-old_rw[i])/abs(new_rw[i]))
        d=compute_price_diff(p, d0)
        manager.update_Q(action_m,rm,lr,d0,d)
        m_reward += sum(rm)
        d0=d
        if env.value[-1]>=TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = (%f). Total Value = %f"
                  % (episode, t, m_reward, env.value[-1]))

            # fig2=plt.figure(2)
            # plt.hist(manager_actions)
            # plt.title('Manager Actions - 0:R, 1:L, 2:S')
            # plt.draw()
            # plt.pause(.01)

            # fig3 = plt.figure(3)
            # plt.plot(worker_actions)
            # plt.title('Worker Actions - 0:S, 1:B, -1:H')
            # plt.show()

            break
        elif env.value[-1]<=0:
            print("Episode %d terminated after %f time steps with total reward = (%f). Total Value = %f"
                  % (episode, t, m_reward, env.value[-1]))
            break
        elif t >= MAX_T-1:
            print("Episode %d terminated after %f time steps with total reward = (%f). Total Value = %f"
                  % (episode, t, m_reward, env.value[-1]))
            break