''' helpful webpages
https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/OpenAI%20Reinforcement%20Learning%20Agent%20Plays%20Atari%20Games.ipynb
https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
'''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from stockDataset import  StockData
from matplotlib import pyplot as plt
import pandas as pd
from stockAgent import StockAgent


''' Defining variables for testing and training '''
device=torch.device('cpu')
torch.manual_seed(2)

stock_path = None #ToDo: Fill this in with the path to the folder with the stock prices
# Can get stock data here https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3

batch_size=1
lr=.005
t=StockData(stock_path,'train')
tgen=DataLoader(t,batch_size=batch_size,shuffle=True,num_workers=1)
s=StockAgent(3)
s=s.to(device)
epochs=20
optimizer = optim.Adam(s.parameters(), lr=lr)

''' training loop '''
avgLoss=0
for e in range(epochs):
    totalLoss=0
    for batch,label in tgen:
        batch=batch.to(device)
        label=label.to(device)
        l,optimizer=s.train(batch,label,optimizer)
        totalLoss+=l
    avgLoss+=float(totalLoss/len(tgen))
    print("Loss:",float(totalLoss/len(tgen)))
torch.save(s,'stock_model.pt') # save the model weights
print('Avg Reward:',torch.mean(torch.tensor(s.reward)))

print()


''' test loop '''
v=StockData(stock_path,'test')
vgen=DataLoader(v,batch_size=batch_size,shuffle=False,num_workers=1)
pred=[]
real=[]
totalLoss=0
for batch,label in vgen:
    batch = batch.to(device)
    label = label.to(device)
    o,l=s.test(batch,label)
    pred.append(o.squeeze())
    real.append(label.squeeze())
    totalLoss+=l
print('Loss:',float(totalLoss/len(vgen)))


''' Plot the predictions '''
p=[]
r=[]
for i in pred:
    p.extend(i)

for i in real:
    r.extend(i)

plt.plot(r)
plt.plot(p)
plt.legend(['real','prediction'],loc='best')
plt.title("Predictions for "+stock_path.split('/')[-1])
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.show()

diff=pd.DataFrame()
diff['preds']=p
diff['real']=r
diff['diff']=abs(diff['preds']-diff['real'])
plt.semilogy(diff.index,diff['diff'])
plt.title("Difference between Real and Prediction for "+stock_path.split('/')[-1])
plt.ylabel("Absolute Valued Difference in Price")
plt.xlabel("Time Step")
plt.show()

print('Avg Diff:',diff['diff'].mean())

