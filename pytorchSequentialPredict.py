from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from stockDataset import  StockData
from matplotlib import pyplot as plt


class DQN(nn.Module):
    def __init__(self,length):
        '''
        :param length: - size of the windows for the input/test data
        '''
        super(DQN,self).__init__()
        self.lstm1=nn.LSTM(length,length)
        self.lstm2 = nn.LSTM(length, length)
        self.lstm3 = nn.LSTM(length, length)
        self.lstm4 = nn.LSTM(length, length)
        self.lstm5 = nn.LSTM(length, length)
        self.lstm6 = nn.LSTM(length, length)

        self.dense1=nn.Linear(length,length)
        self.dense2 = nn.Linear(length,length)
        self.dense3 = nn.Linear(length,length)
        self.dense4 = nn.Linear(length,length)
        self.dense5 = nn.Linear(length,length)
        self.dense6 = nn.Linear(length, length
                                )
        self.relu=nn.ReLU()

    def forward(self, x):
        x,h = self.lstm1(x)
        x = self.dense1(x)
        x=self.relu(x)

        x,h = self.lstm2(x,h)
        x = self.dense2(x)
        x = self.relu(x)

        x,h = self.lstm3(x,h)
        x = self.dense3(x)
        x = self.relu(x)

        x,h = self.lstm4(x,h)
        x = self.dense4(x)
        x = self.relu(x)

        x,h = self.lstm5(x,h)
        x = self.dense5(x)
        x = self.relu(x)

        x, h = self.lstm6(x, h)
        x = self.dense6(x)
        x = self.relu(x)

        return x


''' Declare variables for training/testing '''
device=torch.device('cpu')
torch.manual_seed(2)

epochs=20
batch_size=1
lr = 0.001                  # learning rate

stock_path = None #ToDo: Fill this in with the path to the folder with the stock prices
# Can get stock data here https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
trainOpen=StockData(stock_path,'train')
train_gen=DataLoader(trainOpen, batch_size=batch_size,shuffle=True,num_workers=2)

policy_model=DQN(3)
policy_model=policy_model.to(device)
optimizer = optim.Adam(policy_model.parameters(), lr=lr)

totalLoss=0
avgLoss=0

''' beginning of the training loop '''
for e in range(epochs):
    for batch,label in train_gen:
        batch=batch.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output=policy_model.forward(batch)
        loss = nn.MSELoss()
        l=loss(output,label)
        l.backward()
        optimizer.step()
        totalLoss+=l.item()
    a = totalLoss/len(train_gen)
    avgLoss += a
    totalLoss=0
    print('Epoch:',e,', Loss:', a)

print('Avg Loss:', avgLoss/epochs)

print

''' Testing Loop '''
testOpen=StockData(stock_path,'test')
test_gen=DataLoader(testOpen, batch_size=batch_size,shuffle=False,num_workers=2)
P=[] #predictions
Y=[] # real values
totalLoss=0
with torch.no_grad():
    for batch, label in test_gen:
        Y.append(label.squeeze())
        batch = batch.to(device)
        label=label.to(device)
        P.append(policy_model.forward(batch).squeeze())
        loss = nn.MSELoss()
        l = loss(label,output)
        totalLoss += l.item()
    a = totalLoss / len(test_gen)
    print('Testing Loss:', a)


''' Plot the results'''
p=[]
y=[]
for x in P:
    p.extend(x)

for x in Y:
    y.extend(x)

plt.plot(p)
plt.plot(y)
plt.legend(['predictions','real'])
plt.title('Real Open Price vs Predicted Open Price')
plt.xlabel('Time Step')
plt.ylabel('Price in Dollars')
plt.show()

