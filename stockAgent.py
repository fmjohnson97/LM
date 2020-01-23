import torch
import torch.nn as nn

class StockAgent(nn.Module):
    def __init__(self,length):
        '''
        :param length: - size of the windows for the input/test data
        '''
        super(StockAgent, self).__init__()
        # keep track of the environment
        self.reward=[]
        self.action=[]
        self.loss = nn.MSELoss()
        self.offset=torch.tensor(.01)

        # define the model
        self.lstm1 = nn.LSTM(length, length)
        self.dense1 = nn.Linear(length, length)
        self.lstm2 = nn.LSTM(length, length)
        self.dense2 = nn.Linear(length, length)
        self.lstm3 = nn.LSTM(length, length)
        self.dense3 = nn.Linear(length, length)
        self.lstm4 = nn.LSTM(length, length)
        self.dense4 = nn.Linear(length, length)
        self.lstm5 = nn.LSTM(length, length)
        self.dense5 = nn.Linear(length, length)
        self.lstm6 = nn.LSTM(length, length)
        self.dense6 = nn.Linear(length, length)
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self,x):
        # pass the data through the model
        x, h = self.lstm1(x)
        x = self.relu(x)
        x = self.dense1(x)

        x, h = self.lstm2(x, h)
        x = self.relu(x)
        x = self.dense2(x)

        x, h = self.lstm3(x, h)
        x = self.relu(x)
        x = self.dense3(x)

        x, h = self.lstm4(x, h)
        x = self.relu(x)
        x = self.dense4(x)

        x, h = self.lstm5(x, h)
        x = self.tanh(x)
        x = self.dense5(x)

        return x

    def train(self,x,y,optimizer):
        '''
        :param x: - data to use as input
        :param y: - correct value for the predictions
        :param optimizer: - the optimizer
        :return: - returns loss and the optimizer
        '''
        optimizer.zero_grad()
        output=self.forward(x)
        self.action.append(1+torch.mean(output)*self.offset) #self.action will be the percentage of the original to change by
        newVals=x*self.action[-1]
        l = self.loss(newVals, y)
        self.reward.append(abs(-l.item()))
        l.backward()
        optimizer.step()
        return l.item(), optimizer

    def test(self,x,y):
        '''
        :param x: - data to use as input
        :param y: - correct value for the predictions
        :return: - returns the prediction and the loss
        '''
        with torch.no_grad():
            output = self.forward(x)
            self.action = 1 + torch.mean(output) * self.offset  # self.action will be the percentage of the original to change by
            newVals = x * self.action
            l = self.loss(newVals, y)
            self.reward.append(-l.item())
        return newVals, l.item()

    def clearAllMem(self):
        ''' Resets the reward and action lists'''
        self.reward = []
        self.action = []
