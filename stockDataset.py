import torch
from torch.utils.data import Dataset
import pandas as pd


class StockData(Dataset):
    def __init__(self,file_name,mode=None):
        '''
        :param file_name: - string indicating the path to the text file containing the stock data
                            data can be obtained at the following link
                            https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
        :param mode: - either 'train' or 'test'
        '''

        ''' Get the data from the csv into a pandas dataframe '''
        data=pd.read_csv(file_name, header=0).drop(['Date','Close','High', 'Low', 'Volume', 'OpenInt'], axis=1)
        self.data=data['Open'].tolist()
        self.X=[]#self.data[:-1]
        self.Y=[]#self.data[1:]

        ''' Create the windows of stock open prices '''
        k=0
        while k <len(data)-6:
            self.X.append([self.data[k],self.data[k+1],self.data[k+2]])
            self.Y.append([self.data[k+3],self.data[k+4],self.data[k+5]])
            k+=3

        ''' Seperate the data into test and train '''
        self.Xtrain=self.X[:int(.8*len(self.X))]
        self.Ytrain=self.Y[:int(.8*len(self.X))]
        self.Xtest=self.X[int(.8*len(self.X)):]
        self.Ytest=self.Y[int(.8*len(self.X)):]
        if mode=='test':
            self.X=self.Xtest
            self.Y=self.Ytest
        if mode=='train':
            self.X=self.Xtrain
            self.Y=self.Ytrain

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]).view(1,len(self.X[i])),torch.tensor(self.Y[i]).view(1,len(self.Y[i]))
