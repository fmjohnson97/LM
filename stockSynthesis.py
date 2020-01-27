import glob
import random
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
import numpy as np
from matplotlib import pyplot as plt

''' 1: grab file names from folder and declare train/validation dataframes'''
stock_path = None  # ToDo: Fill this in with the path to the folder with the stock prices
# Can get stock data here https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
files = glob.glob(stock_path)
train = pd.DataFrame()
valid = pd.DataFrame()

''' 2: get data from a random stock price txt file and separate into train and validation'''
k = random.randint(0, len(files))
print(files[k])
data = pd.read_csv(files[k], header=0).drop(['High', 'Low', 'Volume', 'OpenInt'], axis=1)
train = train.append(data.iloc[:int(.8 * len(data)), :])
valid = valid.append(data.iloc[int(.8 * len(data)):, :])

''' 3: filter and reshape train data'''
X = train.filter(['Open']).values
X = X.reshape(X.shape[0], 1, X.shape[1])
y = train['Close'].values.reshape(len(train), 1, 1)

''' 4: set up the LSTM system'''
neurons = 10
batch_size = 1
model = Sequential()
model.add(
    LSTM(neurons, activation='relu', batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(1))
# ToDo: Uncomment these lines to make the network bigger
# model.add(LSTM(neurons,activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
# model.add(Dense(1))
# model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
# model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

''' 5: track performance with tensorboard'''
# tboard=TensorBoard(log_dir="Logs/{}".format(time()))
#ToDo: uncomment this line to use tensorboard to track training performance
'''type  tensorboard --logdir=Logs/  into the command line to track the progress'''

''' 6: train and fit the LSTM '''
epochs = 10
history = model.fit(X, y, epochs=epochs,
                    batch_size=batch_size)  # ,callbacks=[tboard]) #ToDo: uncomment this line if you're using the tensorboard

''' 7: make, plot, and save predictions'''
predictions = []
X = valid.filter(['Open']).values
X = X.reshape(X.shape[0], 1, X.shape[1])
predictions = model.predict(X, batch_size=batch_size)
valid['predictions'] = np.array(predictions).reshape(len(predictions), 1)

plt.plot(data.index, data['Open'], 'c')
plt.plot(range(0, len(data)), data['Close'], 'b')
plt.plot(valid.index, valid['predictions'], 'r')
plt.plot(train.index, train['Close'], 'g')
plt.legend(['Open', 'Close', 'Close Price Predictions'], loc=2)
plt.ylabel('Price')
plt.title('Predicting Close Prices Given Open Prices for ' + files[k].split('/')[-1])
plt.show()
# valid.to_csv('test_results.csv') #ToDo: uncomment this line to save the results to a csv
