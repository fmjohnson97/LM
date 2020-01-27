from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Dropout3d, LSTM, ELU, ReLU, GroupNorm, BatchNorm1d
import cv2
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import glob
import random
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torchvision import transforms

class TSNENet(nn.Module):
    def __init__(self, device, size, getRawData=False, batch=1, mode='udacity'):
        super(TSNENet,self).__init__()
        self.fc1 = Linear(118, 128)
        self.fc2 = Linear(117, 128)
        self.fc3 = Linear(116, 128)
        self.fc4 = Linear(116, 128)
        self.fc5 = Linear(1856, 1024)

        self.conv1=Conv1d(size,32,kernel_size=3, stride=1)
        self.conv2 = Conv1d(32, 32, kernel_size=2, stride=1)
        self.conv3=Conv1d(32,32,kernel_size=2,stride=1)
        self.conv4=Conv1d(32,16,kernel_size=1,stride=1)

        self.fc6=Linear(1024,512)
        self.fc7=Linear(512,256)
        self.fc8=Linear(256,128)
        self.fc9=Linear(256,128)
        self.fc10=Linear(128,10)
        self.lstm1=LSTM(128,128,16)

        self.h1=(torch.rand((16,1,128))/64).to(device)
        self.c1=(torch.rand((16,1,128))/64).to(device)
        self.drop=Dropout3d(.25)
        self.elu=ELU()
        self.relu=ReLU()
        self.laynorm=GroupNorm(1,128)

        self.bnorm1 = BatchNorm1d(32)
        self.bnorm2 = BatchNorm1d(32)
        self.bnorm4 = BatchNorm1d(16)

        self.getRawData=getRawData
        self.batch=batch

    def forward(self,x):
        # import pdb;
        # pdb.set_trace()

        x=self.conv1(x)
        # x=self.bnorm1(x)
        x=self.drop(self.relu(x))
        res1=self.fc1(x[:,-1:,:])

        x=self.conv2(x)
        # x = self.bnorm2(x)
        x=self.drop(self.relu(x))
        res2=self.fc2(x[:,-1:,:])

        x = self.conv3(x)
        # x = self.bnorm2(x)
        x = self.drop(self.relu(x))
        res3 = self.fc3(x[:, -1:, :])

        x=self.conv4(x)
        # x = self.bnorm4(x)
        x=self.drop(self.relu(x))
        res4=self.fc4(x[:,-1:,:])

        x=self.drop(self.relu(self.fc5(x.reshape(x.shape[0],1,-1))))
        x=self.drop(self.relu(self.fc6(x)))
        x=self.drop(self.relu(self.fc7(x)))
        x=self.fc8(x)

        x=self.laynorm(self.elu(x+res1+res2+res3+res4).view(-1,128,1))
        x=x.reshape(x.shape[0],1,-1)
        x=self.drop(x)

        if self.getRawData:
            return x
        else:
            self.h1 = self.h1.detach()
            self.c1 = self.c1.detach()
            out,(h,c)=self.lstm1(x,(self.h1,self.c1))
            self.h1 = h
            self.c1 = c

            out=self.relu(self.fc9(torch.cat((out,x),-1)))
            out=self.fc10(out)
            return out

if __name__=='__main__':
    # read in the image, braking, angle, and throttle data
    path = '/home/faith/Documents/Udacity/Images/'  # None #ToDo: put the path to the csv files for braking, steering, and throttle here
    brakes=pd.read_csv(path+'brake.csv').sort_values('timestamp')
    angles=pd.read_csv(path+'steering.csv').sort_values('timestamp')
    throttle=pd.read_csv(path+'throttle.csv').sort_values('timestamp')
    images=glob.glob(path+'center/*')
    imStamps=[int(x.split('/')[-1].split('.')[0]) for x in images]

    #sort the data into bins to line up the values corresponding to the same timesteps
    AngData = defaultdict(list)
    for i,t in enumerate(angles['timestamp'].astype('int').values):
        AngData[t//(int(1e9)/20)].append(angles['angle'][i])

    ImgData=defaultdict(list)
    for s in imStamps:
        ImgData[s//(int(1e9)/20)].append(s)

    BrakeData = defaultdict(list)
    for i,t in enumerate(brakes['timestamp'].astype('int').values):
        BrakeData[t//(int(1e9)/20)].append(brakes['brake_input'][i])

    ThrotData = defaultdict(list)
    for i,t in enumerate(throttle['timestamp'].astype('int').values):
        ThrotData[t//(int(1e9)/20)].append(throttle['throttle_input'][i])

    # create a list of keys in all data collections for synchronicity
    a_set = set(AngData.keys())
    b_set=set(BrakeData.keys())
    t_set=set(ThrotData.keys())
    i_set = set(ImgData.keys())
    validInds=list(a_set & b_set & i_set & t_set)

    # create the windows of data and the TSNE embedding
    aLabel=[]
    fullData=[]
    i=0
    while i<len(validInds)-40 :
        atemp=[]
        btemp=[]
        ttemp=[]
        aVal=0

        for j in range(i,i+40):
            atemp.append(np.mean(AngData[int(validInds[j])]))
            if abs(atemp[-1])>.07:
                if atemp[-1]>0:
                    aVal+=1
                else:
                    aVal+=-1
            btemp.append(np.mean(BrakeData[int(validInds[j])]))
            ttemp.append(np.mean(ThrotData[int(validInds[j])]))

        aLabel.append(aVal/40)
        atemp.extend(btemp)
        atemp.extend(ttemp)
        fullData.append(atemp)
        i+=40

    tsneA=TSNE()
    data=tsneA.fit_transform(np.array(fullData))

    data=(data-np.min(data,0))/np.max(data-np.min(data,0),0)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    centers=kmeans.cluster_centers_


    experiment =None #ToDo: if you want to use Comet_ML put the experiment initialization code here

    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
    plt.scatter(centers[:,0],centers[:,1],c='r')
    plt.title("Raw Data and Centroids")
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    #initialize the network and other variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=TSNENet(device, 1).to(device)
    opt=torch.optim.Adam(net.parameters(), lr=1e-2)
    loss=nn.CrossEntropyLoss()
    epochs=25
    dataPoint=0
    savedZ=[]
    toTens=transforms.ToTensor()

    #start the training loop
    for e in range(epochs):
        i=0
        vInd=0
        labelInd=0
        print("Epoch:",e)
        while vInd<len(fullData):
            opt.zero_grad()
            indata=torch.tensor(fullData[vInd]).view(1,1,-1)
            Z=net(indata.to(device))
            Z=Z.squeeze(0)
            label=torch.tensor([kmeans.labels_[labelInd]]).long().to(device)
            error=loss(Z,label)
            opt.step()

            # ToDo: uncomment this if you want to track the experiment using comet_ml

            # experiment.log_metric("loss", error.item(), step=dataPoint)
            # experiment.log_metric("Z error",abs(torch.argmax(Z).item()-kmeans.labels_[labelInd]) , step=dataPoint)


            if i%4==3 and i>0:
                labelInd+=1
            i+=1
            vInd += 1
            dataPoint+=1
            if e+1==epochs:
                maxInd=torch.argmax(Z).item()
                coords=centers[maxInd]
                savedZ.append([coords[0],coords[1]])
            if labelInd>=len(kmeans.labels_):
                break


    savedZ=np.array(savedZ)
    plt.scatter(savedZ[:,0],savedZ[:,1],alpha=0.5)
    plt.scatter(centers[:,0],centers[:,1],c='r',marker='X', alpha=.95)
    plt.title("Predicted Centroids")
    plt.legend(['Predicted Points','Real Centroids'])
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    # start the testing loop
    net.eval()
    i = 0
    vInd = 0
    labelInd = 0
    savedZ=[]
    saveInds=[]
    real=[]
    while vInd < len(fullData) and labelInd<len(kmeans.labels_):
        indata = torch.tensor(fullData[vInd]).view(1, 1, -1)
        Z=net(indata.to(device))
        maxInd = torch.argmax(Z).item()
        saveInds.append(maxInd)
        real.append(kmeans.labels_[labelInd])
        coords=centers[maxInd]
        savedZ.append([coords[0],coords[1]])

        if i % 4 == 3 and i > 0:
            labelInd += 1
        i += 1
        vInd+=1



    savedZ=np.array(savedZ)
    plt.scatter(savedZ[:,0],savedZ[:,1])
    plt.title("Predicted Centroids of New Data")
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    plt.scatter(savedZ[:,0],savedZ[:,1],alpha=0.5)
    plt.scatter(centers[:,0],centers[:,1],c='r',marker='X', alpha=.95)
    plt.title("Predicted Centroids")
    plt.legend(['Predicted Points','Real Centroids'])
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    print(confusion_matrix(saveInds,real))

