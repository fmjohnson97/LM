#https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.svm import SVC
from sklearn.decomposition import PCA

''' Make your own data generator for CSVs'''
''''''
stocks=glob.glob("Stocks/*")
etfs=glob.glob('ETFs/*')
'''
#sNums=random.sample(range(len(stocks)),2500)
#eNums=random.sample(range(len(etfs)),1000)

trainX=pd.DataFrame()
trainY=[]
testX=pd.DataFrame()
testY=[]

#for k in sNums[:2000]:
ks=0
while len(trainX.columns)<800 and ks<len(stocks):
    temp = pd.read_csv(stocks[ks], header=0).drop(['Date', 'Close', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)
    if len(temp)>=1000:
        print stocks[ks]
        temp.columns=[stocks[ks].split('/')[-1]]
        trainX=pd.concat([trainX,temp],axis=1)
    ks+=1
    #trainY.append(0)
print
print 'Total train stocks:',len(trainX.columns)
print

ke=0
#for k in eNums[:800]:
while len(trainX.columns)<1600 and ke<len(etfs):
    temp=pd.read_csv(etfs[ke], header=0).drop(['Date','Close','High', 'Low', 'Volume', 'OpenInt'], axis=1)
    if len(temp)>=1000:
        print etfs[ke]
        temp.columns = [etfs[ke].split('/')[-1]]
        trainX =pd.concat([trainX,temp],axis=1)
    ke+=1
    #trainY.append(1)

print
print 'Start test indexes at ks',ks,' and ke ',ke
print

ks=1218
ke=973
#for k in sNums[2000:]:
while len(testX.columns)<300 and ks<len(stocks):
    temp = pd.read_csv(stocks[ks], header=0).drop(['Date', 'Close', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)
    if len(temp)>=1000:
        print stocks[ks]
        temp.columns = [stocks[ks].split('/')[-1]]
        testX =pd.concat([testX,temp],axis=1)
    ks+=1
    #testY.append(0)
print
print 'Total test stocks:',len(testX.columns)
print
#for k in eNums[800:]:
while len(testX.columns)<600 and ke<len(etfs):
    temp = pd.read_csv(etfs[ke], header=0).drop(['Date', 'Close', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)
    if len(temp)>=1000:
        print etfs[ke]
        temp.columns = [etfs[ke].split('/')[-1]]
        testX =pd.concat([testX,temp],axis=1)
    ke+=1
    #testY.append(1)

#print len(trainX.columns)
print len(testX.columns)
#trainX.to_csv('classTrain1000.csv')
testX.to_csv('classTest1000.csv')
'''
''''''
trainX=pd.read_csv('classTrain1000.csv',index_col=0,dtype=np.float64).truncate(before=0, after=999)#.fillna(method='ffill')
testX=pd.read_csv('classTest1000.csv',index_col=0,dtype=np.float64).truncate(before=0, after=999)#.fillna(method='ffill')
X=trainX.T.values
X=X.reshape(X.shape[0],1,X.shape[1])
e=[x.split('/')[-1] for x in etfs]
s=[x.split('/')[-1] for x in stocks]
y=np.array([int(x in e) for x in trainX.columns]).reshape(len(X),1,1)

epochs=20
batch_size=16
neurons=100

''' '''
model = Sequential()
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(LSTM(neurons, activation='relu',batch_input_shape=(batch_size,X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X,y,epochs=epochs,batch_size=batch_size,shuffle=True)

'''
pca=PCA(n_components=100)
pcaX=pca.fit_transform(X)


model=SVC(gamma='auto',kernel='linear')
model.fit(pcaX,y)
'''
''''''
d=len(trainX)-len(testX)
#testX=pd.concat([testX,pd.DataFrame([0]*d,columns=['test'])])
#testX=testX.drop('test',axis=1).fillna(method='ffill')
X=testX.T.values
#pcaX=pca.fit_transform(X)
X=X.reshape(X.shape[0],1,X.shape[1])
y=np.array([int(x in e) for x in testX.columns]).reshape(len(X),1,1)
predictions=model.predict(X,batch_size=batch_size)
#predictions=model.predict(X)
predictions=np.array([round(abs(x)) for x in predictions]).reshape(len(predictions),1)
#predictions=np.array(predictions).reshape(len(predictions),1)
p=pd.DataFrame(predictions)
print(p[0].value_counts())
