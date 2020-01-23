import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import seed

''' Get the data and set the random seed '''
seed(2)
stock_path = None #ToDo: Fill this in with the path to the folder with the stock prices
# Can get stock data here https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
data = pd.read_csv(stock_path, header=0).drop(['Date','Close','High', 'Low', 'Volume', 'OpenInt'], axis=1)

''' create the data windows and create the train and test split'''
groupVals=[]
Y=[]
stop=3 #length of the window
k=0
#Create the windows
while k <len(data)-stop*2-11:
    groupVals.append([data['Open'][k],data['Open'][k+1],data['Open'][k+2]])
    Y.append([data['Open'][k+3],data['Open'][k+4],data['Open'][k+5]])
    k+=stop
#split the data into train data
X=np.array(groupVals[:int(.8*len(groupVals))])
X=X.reshape(X.shape[0],X.shape[1],1)
y=np.array(Y[:int(.8*len(groupVals))])
y=y.reshape(y.shape[0],y.shape[1],1)


''' Define and train the model '''
neurons=100
epochs=10
batch_size=1

model=Sequential()
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons,activation='relu', batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons,activation='relu', batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(X,y,epochs=epochs,batch_size=batch_size,verbose=2)

''' Make predictions '''
predictions=[]
X=np.array(groupVals[int(.8*len(groupVals)):])
y=np.array(Y[int(.8*len(groupVals)):])
X=X.reshape(X.shape[0],X.shape[1],1)
predictions=model.predict(X,batch_size=batch_size)
predictions=np.array(predictions)

''' flatten and plot the results '''
preds=[]
for p in predictions:
  preds.extend(p)

real=[]
for r in y:
  real.extend(r)

avgPrice=[real[0],real[1],real[2]]
for i in range(3,len(real)):
  avgPrice.append((real[i]+real[i-1]+real[i-2])/3.0)

plt.figure(figsize=(10, 7))
plt.plot(real)
plt.plot(preds)
plt.plot(avgPrice)
plt.legend(['Real','Predictions','Avg of Last 3'])
plt.title('Predicting a Sequence of Open Prices Based on a Sequence - zx.us.txt, rmse - '+str(round(sqrt(history.history['loss'][-1]),6)))
plt.ylabel('Price')
plt.xlabel("Time Step")
plt.show()

diff=pd.DataFrame()
diff['preds']=preds
diff['real']=real
diff['diff']=abs(diff['preds']-diff['real'])
plt.semilogy(diff.index,diff['diff'])
plt.title("Difference between Real and Prediction")
plt.ylabel("Absolute Valued Difference")
plt.xlabel("Time Step")
plt.show()

print('Avg Diff:',diff['diff'].mean()[0])
print('Loss:',sqrt(history.history['loss'][-1]))