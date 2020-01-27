from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Conv3d, Conv2d, Dropout3d, Linear, LSTM, ELU, ReLU, GroupNorm
import numpy as np
from udacityData import UdacityData
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

class Challenge(nn.Module):
    def __init__(self, device, size, getRawData=False, mode='udacity'):
        super(Challenge,self).__init__()
        if mode=='udacity':
            self.fc1 = Linear(8295, 128)
            self.fc2 = Linear(1938, 128)
            self.fc3 = Linear(408, 128)
            self.fc4 = Linear(4480, 128)
            self.fc5 = Linear(4480, 1024)
        else:
            self.fc1 = Linear(6195, 128)
            self.fc2 = Linear(1428, 128)
            self.fc3 = Linear(288, 128)
            self.fc4 = Linear(2560, 128)
            self.fc5 = Linear(2560, 1024)
        self.conv1=Conv3d(size,64,kernel_size=(3,12,12), stride=(1,6,6))
        self.conv2=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))
        self.conv3=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))
        self.conv4=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))

        self.fc6=Linear(1024,512)
        self.fc7=Linear(512,256)
        self.fc8=Linear(256,128)
        self.fc9=Linear(258,1)
        self.lstm1=LSTM(130,128,32)
        
        self.h1=torch.zeros(32,1,128).to(device)
        self.c1=torch.zeros(32,1,128).to(device)
        self.drop=Dropout3d(.25)
        self.elu=ELU()
        self.relu=ReLU()
        self.laynorm=GroupNorm(1,128)
        self.getRawData=getRawData

    def forward(self,x,prevOut, ang=None):
        # import pdb;
        # pdb.set_trace()

        x=self.conv1(x)
        x=self.drop(x)
        res1=self.fc1(x[:,-1:,:,:,:].view(1,1,x.shape[-1]*x.shape[-2]))

        x=self.conv2(x.view(1,64,x.shape[-2],x.shape[-1]))
        x=self.drop(x)
        res2=self.fc2(x[:,-1:,:,:].view(1,1,x.shape[-1]*x.shape[-2]))

        x=self.conv3(x)
        x=self.drop(x)
        res3=self.fc3(x[:,-1:,:,:].view(1,1,x.shape[-1]*x.shape[-2]))

        x=self.conv4(x)
        x=self.drop(x)
        res4=self.fc4(x.reshape(1,1,-1))

        x=self.drop(self.relu(self.fc5(x.reshape(1,1,-1))))
        x=self.drop(self.relu(self.fc6(x)))
        x=self.drop(self.relu(self.fc7(x)))
        x=self.fc8(x)

        x=self.laynorm(self.elu(x+res1+res2+res3+res4).view(1,128,1))
        x=x.reshape(1,1,x.shape[-1]*x.shape[-2])
        x=self.drop(x)

        if self.getRawData:
            return x
        else:
            self.h1 = self.h1.detach()
            self.c1 = self.c1.detach()
            out,(h,c)=self.lstm1(torch.cat((prevOut,x),-1),(self.h1,self.c1))#
            self.h1 = h
            self.c1 = c

            out=self.fc9(torch.cat((out,prevOut,x),-1))#
            return out

if __name__ == '__main__':
    #initialize variables and the network
    experiment = None #ToDo: if you want to use Comet_ML put the experiment initialization code here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = None #ToDo: if you want to save the model, put the path here
    kPred=Challenge(device,10).to(device)
    kPred.train()
    opt=optim.Adam(kPred.parameters(), lr=1e-3)
    loss = nn.MSELoss()#nn.L1Loss()#

    #start training
    dataPoint=0
    for change in ['none', 'horiz']:
        uData = UdacityData(change)
        uLoader = DataLoader(uData, batch_size=1, shuffle=False)
        prevOut=torch.zeros(1,1,2).to(device)
        for i, (image, angle) in tqdm(enumerate(uLoader)):
            opt.zero_grad()
            image = image.view(1, 10, 3, 480, 640)
            predAng = kPred(image.to(device),prevOut)

            error = loss(angle.to(device), predAng.view(1))
            error.backward()
            opt.step()

            #use the commented section in the next line to have the previous angle be the predicted angle instead of the ground truth
            prevOut = torch.cat((torch.zeros(1,1,1).to(device),angle.reshape(1,1,1).to(device)),-1)#predAng.detach()),-1)#

            #ToDo: uncomment the next two lines if you're using comet ml

            # experiment.log_metric("loss", error.item(), step=dataPoint)
            # experiment.log_metric("Predicted Angle", predAng.item(), step=dataPoint)
            dataPoint += 1
            if model_save_path is not None and i % 200 == 0 and i > 0:
                torch.save(kPred.state_dict(), model_save_path)

    if model_save_path is not None:
        torch.save(kPred.state_dict(), model_save_path)


    #start the test loop
    kPred.eval()
    uData = UdacityData('none','test')
    uLoader = DataLoader(uData, batch_size=1, shuffle=False)

    preds=[]
    angs=[]
    prevOut=torch.zeros(1,1,2).to(device)
    for i, (image,angle) in tqdm(enumerate(uLoader)):
        image = image.view(1, 10, 3, 480, 640)
        predAng = kPred(image.to(device), prevOut)
        preds.append(predAng.item())

        # use the commented section in the next line to have the previous angle be the predicted angle instead of the ground truth
        prevOut = torch.cat((torch.zeros(1,1,1).to(device),angle.reshape(1,1,1).to(device)),-1)#predAng.detach()),-1)#
        angs.append(angle.item())
        if i%40==0 and i>0:
            plt.plot(angs[i-40:],marker='o')
            plt.plot(preds[i-40:],marker='o')
            plt.title('Predicted vs Real Angle')
            plt.ylabel('Angle in Radians')
            plt.legend(['Real', 'Predicted'])
            plt.xlabel('Timestep')
            plt.show()
            # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
            # experiment.log_figure(figure=plt)
            # plt.clf()

        if i>=500:
            break

