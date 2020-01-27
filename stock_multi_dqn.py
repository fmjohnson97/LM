from multi_stock_env_dqn import MultiStockDQN
from stock_dqn_models import Worker, Manager, Replay
import torch
from matplotlib import pyplot as plt
import numpy as np

NUM_EPISODES = 50000
STREAK_TO_END = 50
MAX_T = 10000
TERMINAL_VALUE = 1000
NUM_SECTORS = 6


### Begin the Simulation ###
env=MultiStockDQN(NUM_SECTORS)
torch.manual_seed(30)
update_target=15
manager=Manager(NUM_SECTORS)
worker = Worker(NUM_SECTORS)
m_replay = Replay()
w_replay = Replay()
fig=plt.figure()
plt.suptitle('Feudal DQN Agent')
ax=fig.add_subplot(111)

for episode in range(NUM_EPISODES):
    # setting the initial learning rates
    lrm = manager.get_learning_rate(episode)
    lrw = worker.get_learning_rate(episode)
    # setting the initial exploration rates
    erm = manager.get_explore_rate(episode)
    erw = worker.get_explore_rate(episode)

    # resetting the manager/worker environments
    p0 = env.reset()
    p0 = np.array(p0).T.tolist()
    m_reward = 0
    w_reward = 0
    m_loss=0
    w_loss=0

    for t in range(MAX_T):
        #select an action
        action_m = manager.select_action(p0,erm)

        # pass goal to the worker
        worker.goal= action_m
        old_val = env.value[-1]

        # make the worker move multiple times per manager instr
        for _ in range(1):

            #select and action
            action_w = worker.select_action(p0, erw)
            #execute the action and get the new prices and state
            p, rw=env.step(action_w)
            w_reward+=sum(rw)
            p = np.array(p).T.tolist()
            w_replay.store(p0, p, action_w, rw)
            #update the q table
            w_loss+=worker.updateDQN(w_replay)

            # render the worker graph
            env.render(ax)
            p0 = p
            if episode % update_target == 0:
                worker.updateTarget()

            if env.value[-1] >= TERMINAL_VALUE:
                break

        # update the q table
        new_val = env.value[-1]
        rm = (new_val - old_val) / new_val
        m_replay.store(p0, p, action_m, rm)
        m_loss+=manager.updateDQN(m_replay)
        m_reward+=rm

        if episode%update_target==0:
            manager.updateTarget()

        if env.value[-1]>=TERMINAL_VALUE:
            print("Episode %d finished after %f time steps with total reward = (%f, %f), loss = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, m_loss, w_loss, env.value[-1]))
            break
        elif env.value[-1]<=0:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f), loss = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, m_loss, w_loss, env.value[-1]))
            break
        elif t >= MAX_T-1:
            print("Episode %d terminated after %f time steps with total reward = (%f, %f), loss = (%f, %f). Total Value = %f"
                  % (episode, t, m_reward, w_reward, m_loss, w_loss, env.value[-1]))
            break