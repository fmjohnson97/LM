from comet_ml import Experiment
from komanda import Challenge
import torch
import torch.nn as nn
from torch.nn import Conv1d, Dropout, Linear, LSTM, ELU, ReLU, GroupNorm
import torch.optim as optim
from torch.utils.data import DataLoader
from udacityData import UdacityData
from tqdm import tqdm
from matplotlib import pyplot as plt

class zModel(nn.Module):
    def __init__(self, device):
        super(zModel,self).__init__()
        self.conv1=Conv1d(1,16,kernel_size=1, stride=1)
        self.conv2=Conv1d(16,16,kernel_size=2,stride=2)
        self.conv3=Conv1d(16,16,kernel_size=3,stride=2)
        self.fc1=Linear(10,32)
        self.fc2=Linear(5,32)
        self.fc3=Linear(2,32)
        self.fc4=Linear(32,128)
        self.fc5=Linear(128,64)
        self.fc6=Linear(64,32)
        self.fc7=Linear(32,1)
        self.lstm1=LSTM(32,16,32)
        self.h1=torch.zeros(32,1,16).to(device)
        self.c1=torch.zeros(32,1,16).to(device)
        self.drop=Dropout(.1)
        self.elu=ELU()
        self.relu=ReLU()
        self.laynorm=GroupNorm(1,32)

    def forward(self,x):
        # import pdb; pdb.set_trace()
        x=self.conv1(x)
        x=self.drop(x)
        res1=self.fc1(x[:,-1:,:])

        x=self.conv2(x)
        x=self.drop(x)
        res2=self.fc2(x[:,-1:,:])

        x=self.conv3(x)
        x=self.drop(x)
        res3=self.fc3(x[:,-1:,:])

        x=self.drop(self.relu(self.fc4(x.view(1,1,-1))))
        x=self.drop(self.relu(self.fc5(x)))
        x=self.fc6(x)

        # import pdb;
        # pdb.set_trace()
        x=self.laynorm(self.elu(x+res1+res2+res3).view(1,32,1))
        x=self.drop(x)

        self.h1 = self.h1.detach()
        self.c1 = self.c1.detach()
        out,(h,c)=self.lstm1(x.view(1,1,-1),(self.h1,self.c1))
        self.h1 = h
        self.c1 = c

        out=self.fc7(x.view(1,1,-1))

        return out

if __name__=='__main__':
    experiment = None #ToDo: if you want to use Comet_ML put the experiment initialization code here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the networks and other things
    ang_save_path = None #ToDo: put the path to save the angle network here; or leave as None to not save it
    z_save_path =  None #ToDo: put the path to save the subroutine ID network here; or leave as None to not save it
    angPred=Challenge(device,10).to(device)
    angPred.train()
    optA=optim.Adam(angPred.parameters(), lr=1e-4)

    zPred=zModel(device).to(device)
    zPred.train()
    optZ=optim.Adam(zPred.parameters(), lr=5e-5)

    loss = nn.L1Loss()#nn.MSELoss()#

    #start the training
    dataPoint = 0
    for change in ['none','horiz','none','horiz','none','horiz','none','horiz','none','horiz','none','horiz','none','horiz','none','horiz']:
        uData = UdacityData(change,'train',10)
        uLoader = DataLoader(uData, batch_size=1, shuffle=False)
        prevOut = torch.rand((1, 1, 1)).to(device)
        angs=[0]*9
        preds=[]
        for i, (image, angle) in tqdm(enumerate(uLoader)):
            optA.zero_grad()
            optZ.zero_grad()

            image = image.view(1, 10, 3, 480, 640)
            angs.append(angle.item())

            predZ=zPred(torch.FloatTensor(angs).reshape(1,1,10).to(device))
            predAng = angPred(image.float().to(device), torch.cat((predZ,prevOut),-1))

            error = loss(angle.to(device), predAng.view(1))
            error.backward()
            optA.step()
            optZ.step()

            angs.pop(0)

            # Currently, this uses the previous predicted angle; use the commented part to have the previous angle be ground truth
            prevOut=predAng.detach()#angle.reshape(1,1,1).to(device)#

            #ToDo: uncomment this if you want to track the experiment using comet_ml

            # experiment.log_metric("loss", error.item(), step=dataPoint)
            # experiment.log_metric("Predicted Angle", predAng.item(), step=dataPoint)
            # experiment.log_metric("Z", predZ.item(), step=dataPoint)
            dataPoint += 1
            if i % 200 == 0 and i > 0:
                if ang_save_path is not None:
                    torch.save(angPred.state_dict(), ang_save_path)
                if z_save_path is not None:
                    torch.save(zPred.state_dict(), z_save_path)

    if ang_save_path is not None:
        torch.save(angPred.state_dict(), ang_save_path)
    if z_save_path is not None:
        torch.save(zPred.state_dict(), z_save_path)


   # start the test loop
    angPred.eval()
    zPred.eval()

    uData = UdacityData('none','test',10)
    uLoader = DataLoader(uData, batch_size=1, shuffle=False)
    angs=[0]*10
    prevOut = torch.rand((1, 1, 1)).to(device)
    a=[]
    z=[]
    new_z=[]
    real=[]

    for i, (image, angle) in tqdm(enumerate(uLoader)):

        image = image.view(1, 10, 3, 480, 640)

        predZ = zPred(torch.tensor(angs).reshape(1,1,10).float().to(device))
        z.append(predZ.item())
        if z[-1]<-.75:
            new_z.append(-1)
        elif z[-1]>.75:
            new_z.append(1)
        else:
            new_z.append(0)
        predAng = angPred(image.float().to(device), torch.cat((predZ,prevOut),-1))
        a.append(predAng.item())
        real.append(angle.item())

        angs.pop(0)
        angs.append(predAng.item())
        # Currently, this uses the previous predicted angle; use the commented part to have the previous angle be ground truth
        prevOut=predAng.detach()#angle.reshape(1,1,1).to(device)#

        if i % 50 == 0 and i > 0:

            # plt.subplot(2,1,1)
            plt.title('Predicted vs Real Angle')#'Co-Training Prediction Results')
            plt.plot(real[i-40:], marker='o')
            plt.plot(a[i-40:], marker='o')
            plt.ylabel('Angle in Radians')
            plt.legend(['Real','Predicted'])
            plt.xlabel('Timestep')
            plt.show()
            # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
            # experiment.log_figure(figure=plt)
            # plt.clf()

            # plt.subplot(2,1,2)
            plt.title('Subroutine ID')
            plt.plot(range(len(z[i-40:])),z[i-40:], marker='o')
            # plt.plot(range(len(new_z[i-40:])), new_z[i-40:], marker='o')
            plt.ylabel('Z Value')
            plt.xlabel('Timestep')
            plt.show()
            # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
            # experiment.log_figure(figure=plt)
            # plt.clf()

        if i>400:
            break
