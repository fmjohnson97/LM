import gym
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import quandl
from datetime import date
from gym import spaces

INIT_BAL = 500
TRANS_FEE = 0.00
Sector_Titles=['Tech','Energy','Finance','Healthcare','Utilities','Transportation']
# ToDo: also listed below are some other stocks to try for each sector
Stocks_Per_Sector={
    'Tech':['ADBE'], # |,'HLIT','INTU',,,'POWI',,'SNX','CPSI','SEAC','SYNA','MSFT','BRKS','MCHP'
    'Energy':['COG'], # |,'NGS',,'PTEN',,'PLUG','CRZO','CPST','CNX','DO','OIS','DRQ','ISRL'
    'Finance':['BANF'], # |,'PFG',,'CACC','GS','EEFT','CPSS','STFC','CASH','FISI',,'CME','SAFT','COLB'
    'Healthcare':['AMGN'], # |,'MYGN',,'PDLI','ICUI','XRAY','GERN',,'AGEN','EXAS','CNMD','GILD','SGEN'
    'Utilities':['ADTN'], # |,'CWST','NTGR','USM',,'WSTL','PLT','RSG','CIEN'
    'Transportation':['HTLD'], #|'ODFL',,,'USAK','CHRW','PTSI','WERN','JBLU','HUBG','UPS','LSTR','SKYW',
}
apiKey = None #ToDo: in order to use quandl you need an API key, so put that here

class MultiTransStock(gym.Env):
    def __init__(self, num_sectors):
        self.action_space=spaces.Discrete(3)
        self.sectors=np.random.choice(range(6), num_sectors, replace=False)#np.array([0,1,2])#
        self.start_date = date(1995, 1, 1)
        self.end_date = date(1996, 1, 1)
        self.stock_ind=0
        self.balance = INIT_BAL
        self.portfolio = [0] * num_sectors
        self.last_price=[1.0]* num_sectors
        self.value = []
        self.state=None
        self.reset()

    def reset(self):
        self.state=pd.DataFrame()
        # uncomment the starred lines to stop the dates from repeating when the environment resets
        # if self.start_date>date.today(): *
        self.start_date = date(1995, 1, 1)
        self.end_date = date(1996, 1, 1)
        # else: *
        #     self.start_date=self.end_date *
        #     self.end_date=self.end_date.replace(year=self.end_date.year+1) *

        # get the stock data for each sector
        for s in self.sectors:
            self.state = pd.concat([self.state, quandl.get('WIKI/' + Stocks_Per_Sector[Sector_Titles[s]][0],
                                                           trim_start=str(self.start_date),
                                                           trim_end=str(self.end_date),
                                                           authtoken=apiKey).filter(['Open'])], axis=1)
            self.state=self.state.dropna()
        self.state.columns = self.sectors

        self.stock_ind = 1
        self.balance=INIT_BAL
        self.portfolio=[0]*len(self.sectors)
        self.last_price = [1.0] * len(self.sectors)
        self.value=[1,1]
        return self.state.values.tolist()[0]

    def render(self, ax):
        #plot the portfolio value
        ax.clear()
        ax.plot(self.value[2:])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio Value in Dollars')
        plt.draw()
        plt.pause(.0000001)

    def get_more_prices(self):
        # get the next stock prices; if the state date > today then it starts the dates over again
        if self.end_date > date.today():
            self.start_date = date(1995, 1, 1)
            self.end_date = date(1996, 1, 1)
            print("Dates have started over!!!!")
        else:
            self.start_date = self.end_date
            self.end_date = self.end_date.replace(year=self.end_date.year + 1)
        self.state = pd.DataFrame()
        for s in self.sectors:
            self.state = pd.concat([self.state, quandl.get('WIKI/' + Stocks_Per_Sector[Sector_Titles[s]][0],
                                                           trim_start=str(self.start_date),
                                                           trim_end=str(self.end_date),
                                                           authtoken=apiKey).filter(['Open'])], axis=1)
            self.state=self.state.dropna()
        self.state.columns = self.sectors
        self.stock_ind = 0

    def step(self, action_m, action_w):
        # get the next stock prices, execute the worker's actions, and compute the respective rewards
        prices = self.state.values[self.stock_ind]
        rw=[]
        old_val=self.compute_value(prices,self.portfolio)
        old_port= self.portfolio
        for i, a in enumerate(action_w):
            num_trade=action_m[i]
            if a == 0 and self.portfolio[i] - num_trade >= 0:  # sell
                self.balance += (prices[i]*num_trade - TRANS_FEE)
                self.portfolio[i] -= num_trade
                self.last_price[i] = prices[i]
            elif a == 1 and self.balance - prices[i]*num_trade - TRANS_FEE >= 0:  # buy
                self.balance -= (prices[i]*num_trade + TRANS_FEE)
                self.portfolio[i] += num_trade
                self.last_price[i] = prices[i]


            # else is hold so you do nothing
            # compute the reward if should end
            if (self.balance - prices[i]*num_trade - TRANS_FEE < 0 and a == 1) or (self.portfolio[i] - num_trade < 0 and a == 0):
                r = -1
            else:
                # store old val
                new_val=self.compute_value(prices, old_port, i)
                if a==-1 or new_val==0:
                    r=0
                else:
                    r=(new_val-old_val)/new_val
            rw.append(r)

        self.stock_ind += 1
        self.update_value(prices)


        if self.stock_ind == len(self.state):
            self.get_more_prices()
        return prices, rw

    def update_value(self, prices):
        # update the portfolio value
        total = 0
        for i, p in enumerate(prices):
            total += self.portfolio[i] * p
        self.value.append(total + self.balance)

    def compute_value(self,prices, portfolio, ind=None):
        # compute a new portfolio value without updating the official list
        total=0
        for i, p in enumerate(prices):
            if i==ind:
                total+=self.portfolio[i]*p
            else:
                total += portfolio[i] * p
        return total