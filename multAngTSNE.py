from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Conv3d, Conv2d, Conv1d, LSTM, Dropout, Dropout3d, ELU, GroupNorm
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.manifold import TSNE
import glob
from collections import defaultdict
import random
import cv2
from torchvision import transforms

class Challenge(nn.Module):
    def __init__(self, device, size, outNum, batch=None):
        super(Challenge, self).__init__()
        self.fc1 = Linear(8295, 128)
        self.fc2 = Linear(1938, 128)
        self.fc3 = Linear(408, 128)
        self.fc4 = Linear(4480, 128)
        self.fc5 = Linear(4480, 1024)

        self.conv1 = Conv3d(size, 64, kernel_size=(3, 12, 12), stride=(1, 6, 6))
        self.conv2 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))

        self.fc6 = Linear(1024, 512)
        self.fc7 = Linear(512, 256)
        self.fc8 = Linear(256, 128)
        self.fc9 = Linear(258, outNum)
        self.lstm1 = LSTM(130, 128, 32)

        self.h1 = (torch.rand((32,1,128))/64).to(device)
        self.c1 = (torch.rand((32,1,128))/64).to(device)
        self.drop = Dropout3d(.25)
        self.elu = ELU()
        self.relu = ReLU()
        self.laynorm = GroupNorm(1, 128)

    def forward(self, x, prevOut):
        # import pdb;
        # pdb.set_trace()

        x = self.conv1(x)
        x = self.drop(x)
        res1 = self.fc1(x[:, -1:, :, :, :].view(-1, 1, x.shape[-1] * x.shape[-2]))

        x = self.conv2(x.view(-1, 64, x.shape[-2], x.shape[-1]))
        x = self.drop(x)
        res2 = self.fc2(x[:, -1:, :, :].view(-1, 1, x.shape[-1] * x.shape[-2]))

        x = self.conv3(x)
        x = self.drop(x)
        res3 = self.fc3(x[:, -1:, :, :].view(-1, 1, x.shape[-1] * x.shape[-2]))

        x = self.conv4(x)
        x = self.drop(x)
        res4 = self.fc4(x.reshape(x.shape[0], 1, -1))

        x = self.drop(self.relu(self.fc5(x.reshape(x.shape[0], 1, -1))))
        x = self.drop(self.relu(self.fc6(x)))
        x = self.drop(self.relu(self.fc7(x)))
        x = self.fc8(x)

        x = self.laynorm(self.elu(x + res1 + res2 + res3 + res4).view(-1, 128, 1))
        x = x.reshape(-1, 1, x.shape[-1] * x.shape[-2])
        x = self.drop(x)

        self.h1 = self.h1.detach()
        self.c1 = self.c1.detach()
        out, (h, c) = self.lstm1(torch.cat((prevOut, x), -1), (self.h1, self.c1))  #
        self.h1 = h
        self.c1 = c

        out = self.fc9(torch.cat((out, prevOut, x), -1))  #
        return out

def train(device, epochs, window, outNum, lrAng, experiment):
    angNet_save_path=None  #ToDo: if you want to save the model, put the path here
    img_path= None #ToDo: put the path for the images here

    #define the angle prediction network, optimizer, loss, and helper functions
    angNet = Challenge(device, window, outNum, 1).to(device)
    angNet.train()
    optAng = optim.Adam(angNet.parameters(), lr=lrAng)
    toTens = transforms.ToTensor()
    loss = nn.L1Loss()

    # read in the image, braking, angle, and throttle data
    path = None #ToDo: put the path to the csv files for braking, steering, and throttle here
    brakes = pd.read_csv(path + 'brake.csv').sort_values('timestamp')
    angles = pd.read_csv(path + 'steering.csv').sort_values('timestamp')
    throttle = pd.read_csv(path + 'throttle.csv').sort_values('timestamp')
    images = glob.glob(path + 'center/*')
    imStamps = [int(x.split('/')[-1].split('.')[0]) for x in images]

    #sort the data into bins to line up the values corresponding to the same timesteps
    AngData = defaultdict(list)
    for i, t in enumerate(angles['timestamp'].astype('int').values):
        AngData[t // (int(1e9) / 20)].append(angles['angle'][i])

    ImgData = defaultdict(list)
    for s in imStamps:
        ImgData[s // (int(1e9) / 20)].append(s)

    BrakeData = defaultdict(list)
    for i, t in enumerate(brakes['timestamp'].astype('int').values):
        BrakeData[t // (int(1e9) / 20)].append(brakes['brake_input'][i])

    ThrotData = defaultdict(list)
    for i, t in enumerate(throttle['timestamp'].astype('int').values):
        ThrotData[t // (int(1e9) / 20)].append(throttle['throttle_input'][i])

    # create a list of keys in all data collections for synchronicity
    a_set = set(AngData.keys())
    b_set = set(BrakeData.keys())
    t_set = set(ThrotData.keys())
    i_set = set(ImgData.keys())
    validInds = list(a_set & b_set & i_set & t_set)

    # create the windows of data and the TSNE embedding
    aLabel = []
    fullData = []
    i = 0
    while i < len(validInds) - 10:
        atemp = []
        btemp = []
        ttemp = []
        aVal = 0

        for j in range(i, i + 10):
            atemp.append(np.mean(AngData[int(validInds[j])]))
            btemp.append(np.mean(BrakeData[int(validInds[j])]))
            ttemp.append(np.mean(ThrotData[int(validInds[j])]))

        aLabel.append(aVal / 10)
        atemp.extend(btemp)
        atemp.extend(ttemp)
        fullData.append(atemp)
        i += 10

    tsneA = TSNE()
    data = tsneA.fit_transform(np.array(fullData))

    # start the angle prediction
    dataPoint = 0
    for e in range(epochs):
        print('Epoch',e)
        ind=0
        labelInd=0
        while  ind < len(validInds)-10:
            image = []
            angle=[]
            #create the image cube and get the angle labels
            for j in range(10):
                imageInd = random.choice(ImgData[int(validInds[ind+j])])
                im = cv2.imread( img_path + str(imageInd) + '.jpg')
                image.append(toTens(im))
                angle.append(np.mean(AngData[int(validInds[ind+j])]))
            image = torch.stack(image).view(1,10,3,480,640)
            angle=torch.tensor(angle).to(device)

            #make the predictions and backpropagate the error
            optAng.zero_grad()
            predZ = torch.tensor(data[labelInd]).view(1,1,-1).to(device)
            predAng = angNet(image.to(device), predZ)

            error = loss(predAng.view(-1), angle[-outNum:])
            error.backward()
            optAng.step()

            # experiment.log_metric('loss',error.item(), step=dataPoint) #ToDo: if you want to keep track of the experiment using comet_ml uncomment this line
            dataPoint += 1
            ind+=10
            labelInd+=1
            if angNet_save_path is not None and dataPoint % 200 == 0 and dataPoint > 0:
                torch.save(angNet.state_dict(), angNet_save_path)

    if angNet_save_path is not None:
        torch.save(angNet.state_dict(), angNet_save_path)

    return angNet, validInds, ImgData, AngData, data

def test(device, outNum, angNet, validInds, ImgData, AngData, data):
    img_path = None #ToDo: put the path for the images here
    save_path = None #ToDo: put a path to save the data to here
    savedAng=[]
    savedReal=[]
    savedZ=[]
    ind = 0
    labelInd = 0
    toTens = transforms.ToTensor()
    while ind < len(validInds) - 10:
        image = []
        angle=[]
        # create the image cube and get the angle labels
        for j in range(10):
            imageInd = random.choice(ImgData[int(validInds[ind + j])])
            im = cv2.imread(img_path + str(imageInd) + '.jpg')
            image.append(toTens(im))
            angle.append(np.mean(AngData[int(validInds[ind + j])]))
        image = torch.stack(image).view(1,10,3,480,640)

        #make the predictions
        predZ = torch.tensor(data[labelInd]).view(1,1,-1).to(device)
        predAng = angNet(image.to(device), torch.tensor(predZ).view(1, 1, -1).to(device))
        savedAng.append(predAng.tolist())
        savedReal.append(angle[-outNum:])
        savedZ.append(predZ.tolist())
        ind += 10
        labelInd += 1
        #save the predictions in a csv
        if ind>400 and outNum>1:
            temp=pd.DataFrame()
            temp['angs']=savedAng
            temp['real']=savedReal
            temp['z']=savedZ
            temp.to_csv(save_path+str(datetime.now())+'.csv')
            break
        elif ind >1200:
            temp = pd.DataFrame()
            temp['angs'] = savedAng
            temp['real'] = savedReal
            temp['z'] = savedZ
            temp.to_csv(save_path + str(datetime.now()) + '.csv')
            break
            # import pdb; pdb.set_trace()

    # i=0
    # while i<400:
    #     plt.title('Predicted vs Real Angle')  # 'Co-Training Prediction Results')
    #     plt.plot(savedReal[i: i + 40], marker='o')
    #     plt.plot(savedAng[i: i + 40], marker='o')
    #     plt.ylabel('Angle in Radians')
    #     plt.legend(['Real', 'Predicted'])
    #     plt.xlabel('Timestep')
    #     # plt.show()
    #     experiment.log_figure(figure=plt)
    #     plt.clf()
    #
    #     # plt.subplot(2,1,2)
    #     plt.title('Subroutine ID')
    #     plt.plot(range(len(savedZ[i: i + 40])), savedZ[i: i + 40], marker='o')
    #     plt.ylabel('Z Value')
    #     plt.xlabel('Timestep')
    #     experiment.log_figure(figure=plt)
    #     plt.clf()
    #     i+=40


if __name__ == '__main__':
    experiment = None #ToDo: if you want to use Comet_ML put the experiment initialization code here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### HyperParameters ###
    batch=6
    outNum=2
    window=10
    lrAng=5e-5
    lrZ=5e-5
    epochs=10
    #######################

    angNet, validInds, ImgData, AngData, data = train(device, epochs, window, outNum, lrAng, experiment)
    test(device, outNum, angNet, validInds, ImgData, AngData, data)