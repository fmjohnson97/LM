import glob
import random
from math import sqrt
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
import numpy as np

from matplotlib import pyplot as plt

''' 1: grab file names from folder and declare train/validation dataframes'''
stock_path =  None #ToDo: Fill this in with the path to the folder with the stock prices
# Can get stock data here https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
files = glob.glob(stock_path)
train = pd.DataFrame()
valid = pd.DataFrame()

''' 2: get data from a random txt file and separate into train and validation'''
k = random.randint(0, len(files)-1)
print(files[k])
data = pd.read_csv(files[k], header=0).drop(['Date', 'Close', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)
data['ShiftOpen'] = data['Open'].shift(1).fillna(method='bfill')
train = train.append(data.iloc[:int(.8 * len(data)), :])
valid = valid.append(data.iloc[int(.8 * len(data)):, :])

''' 3: filter and reshape train data'''
X = train['Open'].values
X = X.reshape(X.shape[0], 1, 1)
y = train['ShiftOpen'].values.reshape(len(train), 1, 1)

''' 4: set up the LSTM network '''
neurons = 10
batch_size = 1
'''find the largest batch size possible with the train and test data sizes'''
for i in range(2, 32):
    if len(train) % i == 0 and len(valid) % i == 0:
        batch_size = i
print('Batch Size = ', batch_size)

model = Sequential()
model.add(LSTM(neurons, activation='relu', batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu', batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu', batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(1))
# ToDo: Uncomment these lines to make the network larger
# model.add(LSTM(neurons,activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
# model.add(Dense(1))
# model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
# model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

''' 5: track performance with tensorboard'''
# tboard=TensorBoard(log_dir="Logs/{}".format(time()))
# ToDo: uncomment this line to use tensorboard to track training performance
'''type  tensorboard --logdir=Logs/  into the command line to track the progress'''

''' 6: train and fit the LSTM '''
epochs = 20
history = model.fit(X, y, epochs=epochs, batch_size=batch_size)  # ,callbacks=[tboard]) #ToDo: uncomment this line if you're using the tensorboard

''' 7: make, graph, and save predictions'''
predictions = []
diff = len(valid) % batch_size
if diff != 0:
    valid = valid.append(pd.DataFrame([[0, 0]] * (batch_size - diff), columns=['Open', 'ShiftOpen']))
X = valid['Open'].values
X = X.reshape(X.shape[0], 1, 1)
predictions = model.predict(X, batch_size=batch_size)

valid['predictions'] = np.array(predictions).reshape(len(predictions), 1)
valid = valid.drop(valid.tail(diff).index)
plt.plot(data.index, data['Open'], 'b')
plt.plot(valid.index, valid['predictions'], 'r')
plt.legend(['Open', 'Predictions'], loc=2)
plt.ylabel('Price')
plt.title('Open Price Predictions on nhtc.us.txt\n' + ' - batch=' + str(batch_size) + ' - epoch=' + str(epochs) + ' - rmse= ' + str(sqrt(history.history['loss'][-1])))
plt.show()

valid['DiffOpen'] = abs(valid['Open'] - valid['ShiftOpen'])
valid['ShiftPred'] = valid['predictions'].shift(1).fillna(method='bfill')
valid['DiffPred'] = abs(valid['predictions'] - valid['ShiftPred'])
plt.plot(valid.index, valid['DiffOpen'])
plt.plot(valid.index, valid['DiffPred'])
plt.title("Difference Between Neighboring Open Prices and Predicted Prices")
plt.ylabel('Difference')
plt.xlabel('Time Step')
plt.legend(['Open Difference', 'Prediction Difference'])
plt.show()
# valid.to_csv('test_results.csv') #ToDo: Uncomment this to save the prediction results to a csv
