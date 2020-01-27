import gym
from matplotlib import pyplot as plt
import pandas as pd
from gym import spaces
import numpy as np
import quandl
from datetime import date

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

class StockEnvDQN(gym.Env):

    def __init__(self, num_sectors):
        self.action_space=spaces.Discrete(3)
        self.sectors=np.random.choice(range(6),num_sectors, replace=False)
        self.start_date = date(1995, 1, 1)
        self.end_date = date(1996, 1, 1)
        self.stock_ind=0
        self.state=None
        self.reset()

    def reset(self):
        self.state=pd.DataFrame()#
        #uncomment the starred lines to stop the dates from repeating when the environment resets
        # if self.start_date>date.today(): *
        self.start_date = date(1995, 1, 1)
        self.end_date = date(1996, 1, 1)
        # else: *
        #     self.start_date=self.end_date *
        #     self.end_date=self.end_date.replace(year=self.end_date.year+1) *

        # get the stock data for each sector
        for s in self.sectors:
            self.state=pd.concat([self.state,quandl.get('WIKI/'+Stocks_Per_Sector[Sector_Titles[s]][0],
                                     trim_start=str(self.start_date),
                                     trim_end=str(self.end_date),
                                     authtoken=apiKey).filter(['Open'])], axis=1)
            self.state=self.state.dropna()
        self.state.columns=self.sectors
        self.stock_ind=1
        # return the first three values
        return self.state.values.tolist()[0:3]

    def render(self, ax, data=None):
        # plot the portfolio value data; if data = None then it plots the stock prices for a given year
        ax.clear()
        if data is None:
            for col in self.state.columns:
                ax.plot(self.state[col].values)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Stock Prices')
        else:
            ax.plot(data)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Portfolio Value in Dollars')
        plt.draw()
        plt.pause(.01)


    def get_more_prices(self):
        #get the next stock prices; if the state date > today then it starts the dates over again
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
        self.stock_ind=0

    def step(self):
        # return the next stock prices
        prices=self.state.values[self.stock_ind:self.stock_ind+3]
        self.stock_ind+=1
        if self.stock_ind==len(self.state)-3:
            self.get_more_prices()
        return prices

