#Data from https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/
#https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

import pandas as pd
import glob
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
import numpy as np
from datetime import datetime

files=glob.glob("/Users/godsrockgirl777/Desktop/Secret Secret/Stocks/*")
data=pd.DataFrame()
#train=pd.DataFrame()
#valid=pd.DataFrame()

''' Put loop here'''
for i in range(1):
    #vals = pd.DataFrame()
    #y = pd.DataFrame()
    #pastVals = []
    #Y = []
    k=random.randint(0,len(files)-1)
    print(files[k])
    #temp=pd.read_csv(files[k], header=0).drop(['High','Low','Volume','OpenInt'], axis=1)
    data=data.append(pd.read_csv(files[k], header=0).drop(['High','Low','Volume','OpenInt'], axis=1))
    '''
        1 - started by just putting date/open as inputs and closed at the output
        2 - did date/open/high/low as input and closed as the output
        3 - doing [open1,open2,open3] as input w/ open4 as output
    
    for k in range(len(temp)-3):
        pastVals.append([temp['Open'][k],temp['Open'][k+1],temp['Open'][k+2]])
        Y.append(temp['Open'][k+3])
    vals=vals.append(pd.DataFrame(pastVals,columns=['p1','p2','p3']))
    y=y.append(pd.DataFrame(Y,columns=['p4']))
    train=train.append(pd.concat([vals.iloc[:int(.8*len(vals)),:],y.iloc[:int(.8*len(vals)),:]],axis=1))
    valid = valid.append(pd.concat([vals.iloc[int(.8 * len(vals)):, :], y.iloc[int(.8 * len(vals)):, :]],axis=1))
    '''
''' End loop now you have several stocks worth of data? want to do only for one at a time?'''

''' (not manipulating dates bc doing sequential data instead)
dates=data['Date'].values
dates=np.array([x.split('-') for x in dates])
data['Year']=dates[:,0]
data['Month']=dates[:,1]
data['Day']=dates[:,2]
data=data.drop('Date',axis=1)
'''

train=data.iloc[:int(.8*len(data)),:]
#train.plot()
#plt.show()
valid=data.iloc[int(.8*len(data)):,:]

# moved on and separated the date into y/m/d
#train['Date']=pd.DatetimeIndex(train['Date']).astype(np.int64)/10**9
#validation['Date']=pd.DatetimeIndex(validation['Date']).astype(np.int64)/10**9

''' 1 - predicting on the scaled difference between each entry and the one before it
      - input is date as unix and open price
      - predict the close price
train['history']=pd.DataFrame(train['Close'].shift(1).fillna(0))
validation['history']=pd.DataFrame(validation['Close'].shift(1).fillna(0))
train['diff']=train['Close']-train['history']
validation['diff']=validation['Close']-validation['history']
scaler=MinMaxScaler(feature_range=(-1,1)).fit(train['diff'].values.reshape(-1,1))
scaledDiff=scaler.transform(train['diff'].values.reshape(-1,1))
train['scaledDiff']=scaledDiff
scaledOpen=scaler.transform(train['Open'].values.reshape(-1,1))
train['scaledOpen']=scaledOpen
scaledVOpen=scaler.transform(validation['Open'].values.reshape(-1,1))
validation['scaledOpen']=scaledVOpen
scaledVDiff=scaler.transform(validation['diff'].values.reshape(-1,1))
validation['scaledDiff']=scaledVDiff
'''
''' 1 - tried without scaling anything
    2 - predicting the close price (scaled) based on the scaled open price and date in unix
    3 - predicting close price (scaled) based on scaled open price and date separated by y/m/d
    4 - predicting with a scaled range of 0-1 instead of -1-1
    
scaler=MinMaxScaler(feature_range=(0,1)).fit(train['High'].values.reshape(-1,1))
train['Open']=scaler.transform(train['Open'].values.reshape(-1,1))
#train['High']=scaler.transform(train['High'].values.reshape(-1,1))
#train['Low']=scaler.transform(train['Low'].values.reshape(-1,1))
train['Close']=scaler.transform(train['Close'].values.reshape(-1,1))
validation['Open']=scaler.transform(validation['Open'].values.reshape(-1,1))
validation['Close']=scaler.transform(validation['Close'].values.reshape(-1,1))
#validation['High']=scaler.transform(validation['High'].values.reshape(-1,1))
#validation['Low']=scaler.transform(validation['Low'].values.reshape(-1,1))
'''

''' 1 - put data in straight -> only predicted 1 value; terrible results'''
X=train.filter(['Open']).values
X=X.reshape(X.shape[0],1,X.shape[1])
y=train['Close'].values.reshape(len(train),1,1)

''' 2 - training on sequential price data
X=train.filter(['p1','p2','p3']).values
#X=X.reshape()
y=train['p4'].values.reshape(len(train),1)
'''
print(X.shape)
print(y.shape)

''' 1 - Tried with 1 neuron -> only predicted one number
    2 - tried with 5 neuron -> only predicted one number
    3 - tried with 3 neuron -> only predicted one number
    4 - tried with 50 neurons -> only predicted one number
    5 - 
'''

neurons=50
batch_size=1
model=Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
#model.add(Dense(1))
model.add(LSTM(neurons, batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

epochs=10
''' - tried this; gonna try just a plain fit with epochs
for i in range(epochs):
    model.fit(X,y,epochs=1,batch_size=batch_size,shuffle=False)
    #model.reset_states()
'''

model.fit(X,y,epochs=epochs,batch_size=batch_size)

predictions=[]
X=valid.filter(['Open']).values#valid.filter(['p1','p2','p3']).values
X=X.reshape(X.shape[0],1,X.shape[1])

predictions=model.predict(X, batch_size=batch_size)
valid['predictions']=predictions.reshape(len(predictions),1)

print(valid)#,predictions#validation

valid.to_csv('test_results.csv')



#unScaledDiff=scaler.inverse_transform(scaledDiff)
#train['diff']=train['diff']+train['history']
