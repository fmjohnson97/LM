import quandl
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

# same dataset class as stockDataset.py
class StockData(Dataset):
    def __init__(self,data,mode=None):
        self.data=data
        self.X=[]#self.data[:-1]
        self.Y=[]#self.data[1:]
        k=0
        while k <len(self.data)-1:
            self.X.append(data[k])
            self.Y.append(data[k+1])
            k+=1
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

# same agent from stockAgent.py
class StockAgent(nn.Module):
    def __init__(self,length):
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

        x, h = self.lstm6(x, h)
        x = self.relu(x)
        x = self.dense6(x)
        return x

    def train(self,x,y,optimizer):
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
        with torch.no_grad():
            output = self.forward(x)
            self.action.append(1 + torch.mean(output) * self.offset)  # self.action will be the percentage of the original to change by
            newVals = x * self.action[-1]
            l = self.loss(newVals, y)
            self.reward.append(-l.item())
        return newVals, l.item()

    def clearAllMem(self):
        self.reward = []
        self.action = []

### Get the stock data into a data frame ###

apiKey= None #ToDo: in order to use quandl you need an API key, so put that here

#Technology Sector
#Computer Communications Equipment
jnpr = quandl.get("WIKI/JNPR", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Computer Software: Prepackaged Software
gwre = quandl.get("WIKI/GWRE", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Diversified Commercial Services
kfy = quandl.get("WIKI/KFY", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#EDP Services
epam = quandl.get("WIKI/EPAM", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Industrial Machinery/Components
rxn = quandl.get("WIKI/RXN", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Professional Services
asgn= quandl.get("WIKI/ASGN", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Retail: Computer Software & Peripheral Equipment
snx = quandl.get("WIKI/SNX", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])
#Semiconductors
iphi = quandl.get("WIKI/IPHI", trim_start = "2014-1-1", trim_end = "2018-12-31",authtoken=apiKey).filter(['Open'])

data=pd.DataFrame(epam)
data=data.merge(jnpr,on='Date')
data=data.merge(snx,on='Date')
data=data.merge(gwre,on='Date')
data=data.merge(iphi,on='Date')
data=data.merge(rxn,on='Date')
data=data.merge(asgn,on='Date')
data=data.merge(kfy,on='Date')

data.columns=['epam','jnpr','snx','gwre','iphi','rxn','asgn','kfy']


### Initialize variables for learning

device=torch.device('cuda' if torch.cuda.is_available()  else "cpu")
torch.manual_seed(2)


# train loop for the multiplier agent
batch_size=1
lr=.005
t=StockData(data.values.tolist(),'train')
tgen=DataLoader(t,batch_size=batch_size,shuffle=True,num_workers=1)
s=StockAgent(8)
s=s.to(device)
epochs=20
optimizer = optim.Adam(s.parameters(), lr=lr)
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
print('Avg Reward:',torch.mean(torch.tensor(s.reward)))

print()

#test loop for the multiplier agent
v=StockData(data.values.tolist(),'test')
vgen=DataLoader(v,batch_size=batch_size,shuffle=False,num_workers=1)
pred=[]
real=[]
totalLoss=0
for batch,label in vgen:
    batch = batch.to(device)
    label = label.to(device)
    o,l=s.test(batch,label)
    pred.append(o[0][0])
    real.append(label[0][0])
    totalLoss+=l
print('Loss:',float(totalLoss/len(vgen)))


### unravel the prediction data ###

pj=[]
pg=[]
pk=[]
pe=[]
pr=[]
pa=[]
ps=[]
pi=[]

rj=[]
rg=[]
rk=[]
re=[]
rr=[]
ra=[]
rs=[]
ri=[]

for i in pred:
    i=i.tolist()
    pe.append(i[0])
    pj.append(i[1])
    ps.append(i[2])
    pg.append(i[3])
    pi.append(i[4])
    pr.append(i[5])
    pa.append(i[6])
    pk.append(i[7])

for i in real:
    i=i.tolist()
    re.append(i[0])
    rj.append(i[1])
    rs.append(i[2])
    rg.append(i[3])
    ri.append(i[4])
    rr.append(i[5])
    ra.append(i[6])
    rk.append(i[7])


### Plot the prediction data for the multiplier agent
plt.figure(figsize=(10, 7))
plt.plot(rj,'-.b',label='JNPR')
plt.plot(pj,'-.g',label=None)

plt.plot(rk,'--b',label='KFY')
plt.plot(pk,'--g',label=None)

plt.plot(rr,':b',label='RXN')
plt.plot(pr,':g',label=None)

plt.plot(ri,'b',label='IPHI')
plt.plot(pi,'g',label=None)

plt.legend()
plt.title("Real vs Predicted Open Price")
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(ra,'-.b',label='ASGN')
plt.plot(pa,'-.g',label=None)

plt.plot(rg,'--b',label='GWRE')
plt.plot(pg,'--g',label=None)

plt.plot(re,':b',label='EPAM')
plt.plot(pe,':g',label=None)

plt.plot(rs,'b',label='SNX')
plt.plot(ps,'g',label=None)

plt.legend(['real','prediction'],loc='best')
plt.title("Real vs Predicted Open Price")
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.show()

diff=pd.DataFrame()
diff['pj']=pj
diff['rj']=rj
diff['dj']=abs(diff['pj']-diff['rj'])

diff['pg']=pg
diff['rg']=rg
diff['dg']=abs(diff['pg']-diff['rg'])

diff['pk']=pk
diff['rk']=rk
diff['dk']=abs(diff['pk']-diff['rk'])

diff['pe']=pe
diff['re']=re
diff['de']=abs(diff['pe']-diff['re'])

diff['pr']=pr
diff['rr']=rr
diff['dr']=abs(diff['pr']-diff['rr'])

diff['pa']=pa
diff['ra']=ra
diff['da']=abs(diff['pa']-diff['ra'])

diff['ps']=ps
diff['rs']=rs
diff['ds']=abs(diff['ps']-diff['rs'])

diff['pi']=pi
diff['ri']=ri
diff['di']=abs(diff['pi']-diff['ri'])


plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dj'])
plt.title("Difference between Real and Predicted Price for JNPR")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dg'])
plt.title("Difference between Real and Predicted Price for GWRE")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dk'])
plt.title("Difference between Real and Predicted Price for KFY")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['de'])
plt.title("Difference between Real and Predicted Price for EPAM")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dr'])
plt.title("Difference between Real and Predicted Price for RXN")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['da'])
plt.title("Difference between Real and Predicted Price for ASGN")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['ds'])
plt.title("Difference between Real and Predicted Price for SNX")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['di'])
plt.title("Difference between Real and Predicted Price for IPHI")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

print('Avg Diff jnpr:',diff['dj'].mean())
print('Avg Diff gwre:',diff['dg'].mean())
print('Avg Diff kfy:',diff['dk'].mean())
print('Avg Diff epam:',diff['de'].mean())
print('Avg Diff rxn:',diff['dr'].mean())
print('Avg Diff asgn:',diff['da'].mean())
print('Avg Diff snx:',diff['ds'].mean())
print('Avg Diff iphi:',diff['di'].mean())

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
^^^^ Code for the multiplier agent

vvvv Code for the LSTM network
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### generate the data windows for the LSTM
groupVals=data.values.tolist()
X=[]
Y=[]

k=0
while k <len(groupVals)-1:
    X.append(groupVals[k])
    Y.append(groupVals[k+1])
    k+=1

x=np.array(X[:int(.8*len(groupVals))])
x=x.reshape(x.shape[0],x.shape[1],1)
y=np.array(Y[:int(.8*len(groupVals))])
y=y.reshape(y.shape[0],y.shape[1],1)

print(x.shape)
print(y.shape)

# initialize and train the network
neurons=100
epochs=30
batch_size=1
# for i in range(2,3):
#     if len(X)%i==0 and len(groupVals[int(.8*len(groupVals)):])%i==0:
#         batch_size=i
# print ('Batch Size = ',batch_size)

model=Sequential()
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons,activation='relu', batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons,activation='relu', batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,x.shape[1],x.shape[2]), return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(x,y,epochs=epochs,batch_size=batch_size,verbose=2)

# test loop
predictions=[]
x=np.array(X[int(.8*len(groupVals)):])
y=np.array(Y[int(.8*len(groupVals)):])
x=x.reshape(x.shape[0],x.shape[1],1)
predictions=model.predict(x,batch_size=batch_size)
predictions=np.array(predictions)


#plot the LSTM predictions
pj=[]
pg=[]
pk=[]
pe=[]
pr=[]
pa=[]
ps=[]
pi=[]

rj=[]
rg=[]
rk=[]
re=[]
rr=[]
ra=[]
rs=[]
ri=[]

for i in predictions:
    pe.append(i[0])
    pj.append(i[1])
    ps.append(i[2])
    pg.append(i[3])
    pi.append(i[4])
    pr.append(i[5])
    pa.append(i[6])
    pk.append(i[7])

for i in y:
    re.append(i[0])
    rj.append(i[1])
    rs.append(i[2])
    rg.append(i[3])
    ri.append(i[4])
    rr.append(i[5])
    ra.append(i[6])
    rk.append(i[7])

plt.figure(figsize=(10, 7))
plt.plot(rj,'-.b',label='JNPR')
plt.plot(pj,'-.g',label=None)

plt.plot(rk,'--b',label='KFY')
plt.plot(pk,'--g',label=None)

plt.plot(rr,':b',label='RXN')
plt.plot(pr,':g',label=None)

plt.plot(ri,'b',label='IPHI')
plt.plot(pi,'g',label=None)

#plt.axis([0,215,20,52])
plt.legend()#['JNPR','KFY','RXN','IPHI'])
plt.title("Real vs Predicted Open Price")
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(ra,'-.b',label='ASGN')
plt.plot(pa,'-.g',label=None)

plt.plot(rg,'--b',label='GWRE')
plt.plot(pg,'--g',label=None)

plt.plot(re,':b',label='EPAM')
plt.plot(pe,':g',label=None)

plt.plot(rs,'b',label='SNX')
plt.plot(ps,'g',label=None)

#plt.axis([0,215,42,142])
plt.legend(['real','prediction'],loc='best')
plt.title("Real vs Predicted Open Price")
plt.legend()#['ASGN','GWRE','EPAM','SNX'])
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.show()

diff=pd.DataFrame()
diff['pj']=pj
diff['rj']=rj
diff['dj']=abs(diff['pj']-diff['rj'])

diff['pg']=pg
diff['rg']=rg
diff['dg']=abs(diff['pg']-diff['rg'])

diff['pk']=pk
diff['rk']=rk
diff['dk']=abs(diff['pk']-diff['rk'])

diff['pe']=pe
diff['re']=re
diff['de']=abs(diff['pe']-diff['re'])

diff['pr']=pr
diff['rr']=rr
diff['dr']=abs(diff['pr']-diff['rr'])

diff['pa']=pa
diff['ra']=ra
diff['da']=abs(diff['pa']-diff['ra'])

diff['ps']=ps
diff['rs']=rs
diff['ds']=abs(diff['ps']-diff['rs'])

diff['pi']=pi
diff['ri']=ri
diff['di']=abs(diff['pi']-diff['ri'])


plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dj'])
plt.title("Difference between Real and Predicted Price for JNPR")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dg'])
plt.title("Difference between Real and Predicted Price for GWRE")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dk'])
plt.title("Difference between Real and Predicted Price for KFY")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['de'])
plt.title("Difference between Real and Predicted Price for EPAM")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['dr'])
plt.title("Difference between Real and Predicted Price for RXN")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['da'])
plt.title("Difference between Real and Predicted Price for ASGN")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['ds'])
plt.title("Difference between Real and Predicted Price for SNX")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

plt.figure(figsize=(10, 7))
plt.semilogy(diff.index,diff['di'])
plt.title("Difference between Real and Predicted Price for IPHI")
plt.ylabel("Difference in Price")
plt.xlabel("Time Step")
plt.show()

print('Avg Diff jnpr:',diff['dj'].mean())
print('Avg Diff gwre:',diff['dg'].mean())
print('Avg Diff kfy:',diff['dk'].mean())
print('Avg Diff epam:',diff['de'].mean())
print('Avg Diff rxn:',diff['dr'].mean())
print('Avg Diff asgn:',diff['da'].mean())
print('Avg Diff snx:',diff['ds'].mean())
print('Avg Diff iphi:',diff['di'].mean())
