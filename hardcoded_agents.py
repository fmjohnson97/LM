import numpy as np
import random

INIT_BAL=500

def sampleAction(limit):
    return random.randint(0, limit)

class ShortSellingAgent(object):
    def __init__(self, sectors, sthresh=.5, bthresh=-.5):
        self.sectors = sectors
        self.s_thresh = sthresh
        self.b_thresh = bthresh

    def select_action(self, args):
        state, last_price, er = args
        action = []
        if random.random()<er:
            #for _ in range(self.sectors):
            action.append(sampleAction(1))
        else:
            diff = np.array(state) - np.array(last_price)
            # sell if price goes up; buy if price goes down
            #for i,d in enumerate(diff):
            d=diff
            if d > self.s_thresh:
                action.append(0) #sell
            elif d < self.b_thresh:
                action.append(1) #buy
            else:
                action.append(-1) #hold

        return action[0]

class LongSellingAgent(object):
    def __init__(self, sectors, sthresh=-.5, bthresh=.5):
        self.sectors = sectors
        self.s_thresh = sthresh
        self.b_thresh = bthresh

    def select_action(self, args):
        state, last_price, er = args
        action = []
        if random.random()<er:
            #for _ in range(self.sectors):
            action.append(sampleAction(1))
        else:
            diff = np.array(state) - np.array(last_price)
            # sell if price goes down; buy if price goes up
            #for i,d in enumerate(diff):
            d=diff
            if d < self.s_thresh:
                action.append(0) #sell
            elif d > self.b_thresh:
                action.append(1) #buy
            else:
                action.append(-1) #hold

        return action[0]

class RandomAgent(object):
    def __init__(self, sectors):
        self.sectors = sectors

    def select_action(self, args=None):
        action = []
        #for _ in range(self.sectors):
        a = sampleAction(2)
        if a == 2:
            action.append(-1)
        else:
            action.append(a)

        return action[0]

class AgentManager(object):
    def __init__(self, sectors):
        self.sectors=sectors
        self.discount_factor=.99
        self.q_table=np.zeros((4,3), dtype=float)

    def select_action(self, diff, er):
        action=[] # will give the number agent to use
        for s in range(self.sectors):
            if random.random()<=er:
                action.append(sampleAction(2))
            else:
                action.append(int(np.argmax(self.q_table[diff[s]])))

        return action

    def update_Q(self, action, reward, lr, od, d):
        for i,a in enumerate(action):
            bestQ=np.amax(self.q_table[d])
            self.q_table[od,a]+=lr*(reward[i]+self.discount_factor*bestQ-self.q_table[od,a])
