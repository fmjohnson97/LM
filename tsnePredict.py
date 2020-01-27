from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.nn import Linear, Conv3d, Conv2d, Dropout3d, LSTM, ELU, ReLU, GroupNorm, BatchNorm3d, BatchNorm2d, MaxPool2d, \
    MaxPool3d
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
from scipy.signal import convolve2d


def normIm(im):
    A = [[np.min(im), 1], [np.max(im), 1]]
    B = [[0], [255]]
    q = np.linalg.solve(A, B)
    return im * q[0] + q[1]


class TSNENet(nn.Module):
    def __init__(self, device, size, getRawData=False, batch=1, mode='udacity'):
        super(TSNENet, self).__init__()
        self.fc1 = Linear(8295, 128)  # 8374
        self.fc2 = Linear(475, 128)
        self.fc3 = Linear(88, 128)
        self.fc4 = Linear(512, 128)
        self.fc5 = Linear(512, 1024)

        self.conv1 = Conv3d(size, 64, kernel_size=(3, 12, 12), stride=(1, 6, 6))  # , padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2))

        self.fc6 = Linear(1024, 512)
        self.fc7 = Linear(512, 256)
        self.fc8 = Linear(256, 128)
        self.fc9 = Linear(258, 128)
        self.fc10 = Linear(128, 15)
        self.lstm1 = LSTM(130, 128, 32)

        self.h1 = (torch.rand((32, 1, 128)) / 64).to(device)
        self.c1 = (torch.rand((32, 1, 128)) / 64).to(device)
        self.drop = Dropout3d(.05)
        self.elu = ELU()
        self.relu = ReLU()
        self.laynorm = GroupNorm(1, 128)

        self.bnorm1 = BatchNorm3d(64)
        self.bnorm2 = BatchNorm2d(64)
        self.bnorm3 = BatchNorm2d(64)
        self.bnorm4 = BatchNorm2d(64)

        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)

        self.getRawData = getRawData
        self.batch = batch

    def forward(self, x, prevCen):
        # import pdb;
        # pdb.set_trace()

        x = self.conv1(x)
        # x=self.bnorm1(x)
        x = self.drop(self.relu(x))
        res1 = self.fc1(x[:, -1:, :, :, :].view(x.shape[0], 1, -1))

        x = self.conv2(x.view(-1, 64, x.shape[-2], x.shape[-1]))
        # x = self.bnorm2(x)
        x = self.drop(self.relu(x))
        x = self.pool1(x)
        res2 = self.fc2(x[:, -1:, :, :].view(x.shape[0], 1, -1))

        x = self.conv3(x)
        # x = self.bnorm3(x)
        x = self.drop(self.relu(x))
        res3 = self.fc3(x[:, -1:, :, :].view(x.shape[0], 1, -1))

        x = self.conv4(x)
        # x = self.bnorm4(x)
        x = self.drop(self.relu(x))
        # x = self.pool2(x)
        res4 = self.fc4(x.reshape(x.shape[0], 1, -1))

        x = self.drop(self.relu(self.fc5(x.reshape(x.shape[0], 1, -1))))
        x = self.drop(self.relu(self.fc6(x)))
        x = self.drop(self.relu(self.fc7(x)))
        x = self.fc8(x)

        # x=self.laynorm(self.elu(x+res1+res2+res3+res4).view(-1,128,1))
        # x=x.reshape(x.shape[0],1,-1)
        # x=self.drop(x)

        if self.getRawData:
            return x
        else:
            self.h1 = self.h1.detach()
            self.c1 = self.c1.detach()
            out, (h, c) = self.lstm1(torch.cat((x, prevCen), -1), (self.h1, self.c1))
            self.h1 = h
            self.c1 = c

            out = self.relu(self.fc9(torch.cat((prevCen, out, x), -1)))
            out = self.fc10(out)
            return out


if __name__ == '__main__':
    # Get braking, steering, image, and throttle data
    tsneNet_save_path = None #ToDo: if you want to save the model, put the path here
    img_path =None #ToDo: put the path for the images here
    path = None #ToDo: put the path to the csv files for braking, steering, and throttle here
    brakes = pd.read_csv(path + 'brake.csv').sort_values('timestamp')
    angles = pd.read_csv(path + 'steering.csv').sort_values('timestamp')
    throttle = pd.read_csv(path + 'throttle.csv').sort_values('timestamp')
    images = glob.glob(path + 'center/*')
    imStamps = [int(x.split('/')[-1].split('.')[0]) for x in images]

    # sort the data into bins to line up the values corresponding to the same timesteps
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
            if abs(atemp[-1]) > .07:
                if atemp[-1] > 0:
                    aVal += 1
                else:
                    aVal += -1
            btemp.append(np.mean(BrakeData[int(validInds[j])]))
            ttemp.append(np.mean(ThrotData[int(validInds[j])]))

        aLabel.append(aVal / 10)
        atemp.extend(btemp)
        atemp.extend(ttemp)
        fullData.append(atemp)
        i += 10

    tsneA = TSNE()
    data = tsneA.fit_transform(np.array(fullData))

    # normalize the tsne embedding coordinates
    data = (data - np.min(data, 0)) / np.max(data - np.min(data, 0), 0)

    kmeans = KMeans(n_clusters=15, random_state=0).fit(data)
    centers = kmeans.cluster_centers_

    # ToDo: Uncomment this code to make csv files with the tsne labels and centroids

    # import pandas as pd
    # temp=pd.DataFrame()
    # temp['labels']=kmeans.labels_
    # temp.to_csv('tsneLabels.csv')
    # temp=pd.DataFrame()
    # temp['X']=centers[:,0]
    # temp['Y'] = centers[:, 1]
    # temp.to_csv('tsneCenters.csv')
    # import pdb;pdb.set_trace()

    experiment = None #ToDo: if you want to use Comet_ML put the experiment initialization code here

    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title("Raw Data and Centroids")
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    # Initialize variables and the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TSNENet(device, 10).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    epochs = 10
    dataPoint = 0
    savedZ = []
    toTens = transforms.ToTensor()

    # start training
    for e in range(epochs):
        i = 0
        vInd = 0
        labelInd = 0
        prevCentroid = torch.rand((1, 1, 2)).to(device)
        print("Epoch:", e)
        while vInd < len(validInds) - 10:
            image = []
            for j in range(10):  # 5):
                imageInd = random.choice(ImgData[validInds[vInd + j]])
                im = cv2.imread(img_path + str(imageInd) + '.jpg')
                # ToDo:if you want to use the optical flow images, uncomment the following lines

                # g2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # ix = convolve2d(g2, [[1, 0, -1]], boundary='symm', mode='same')
                # iy = convolve2d(g2, [[1], [0], [-1]], boundary='symm', mode='same')
                # ix = normIm(ix)
                # iy = normIm(iy)
                # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                image.append(toTens(im))
                # image.append(toTens(ix))
                # image.append(toTens(iy))

            image = torch.stack(image)
            opt.zero_grad()
            image = image.view(1, 10, 3, 480, 640)
            Z = net(image.float().to(device), prevCentroid)
            Z = Z.squeeze(0)
            label = torch.tensor([kmeans.labels_[labelInd]]).long().to(device)
            error = loss(Z, label)
            opt.step()
            maxInd = torch.argmax(Z).item()
            prevCentroid = torch.tensor(centers[maxInd]).view(1, 1, -1).to(device)

            # ToDo: if you want to keep track of the experiment using comet_ml uncomment these lines

            # experiment.log_metric("loss", error.item(), step=dataPoint)
            # experiment.log_metric("Z error",abs(torch.argmax(Z).item()-kmeans.labels_[labelInd]) , step=dataPoint)

            # if i%2==1 and i>0:
            labelInd += 1
            i += 1
            vInd += 10
            dataPoint += 1
            if tsneNet_save_path is not None and i % 100 == 0 and i > 0:
                torch.save(net.state_dict(), tsneNet_save_path)
            if e + 1 == epochs:
                maxInd = torch.argmax(Z).item()
                coords = centers[maxInd]
                savedZ.append([coords[0], coords[1]])
            if labelInd >= len(kmeans.labels_):
                break

    if tsneNet_save_path is not None:
        torch.save(net.state_dict(), tsneNet_save_path)
    savedZ = np.array(savedZ)
    plt.scatter(savedZ[:, 0], savedZ[:, 1], alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='X', alpha=.95)
    plt.title("Predicted Centroids")
    plt.legend(['Predicted Points', 'Real Centroids'])
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    # start the testing loop
    net.eval()
    i = 0
    vInd = 0
    labelInd = 0
    prevCentroid = torch.rand((1, 1, 2)).to(device)
    savedZ = []
    saveInds = []
    real = []
    while vInd < len(validInds) - 10 and labelInd < len(kmeans.labels_):
        image = []
        for j in range(10):  # 0):
            imageInd = random.choice(ImgData[validInds[vInd + j]])
            im = cv2.imread(img_path + str(imageInd) + '.jpg')
            # ToDo:if you want to use the optical flow images, uncomment the following lines

            # g2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # ix = convolve2d(g2, [[1, 0, -1]], boundary='symm', mode='same')
            # iy = convolve2d(g2, [[1], [0], [-1]], boundary='symm', mode='same')
            # ix = normIm(ix)
            # iy = normIm(iy)
            # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            image.append(toTens(im))
            # image.append(toTens(ix))
            # image.append(toTens(iy))
        image = torch.stack(image)
        image = image.view(1, 10, 3, 480, 640)

        Z = net(image.float().to(device), prevCentroid)
        maxInd = torch.argmax(Z).item()
        saveInds.append(maxInd)
        real.append(kmeans.labels_[labelInd])
        coords = centers[maxInd]
        savedZ.append([coords[0], coords[1]])

        prevCentroid = torch.tensor(centers[maxInd]).view(1, 1, -1).to(device)

        # if i % 2 == 1 and i > 0:
        labelInd += 1
        i += 1
        vInd += 10

    # graph the predictions
    savedZ = np.array(savedZ)
    plt.scatter(savedZ[:, 0], savedZ[:, 1])
    plt.title("Predicted Centroids of New Data")
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    plt.scatter(savedZ[:, 0], savedZ[:, 1], alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='X', alpha=.95)
    plt.title("Predicted Centroids")
    plt.legend(['Predicted Points', 'Real Centroids'])
    plt.show()
    # ToDo: if you're using comet_ml, comment out the above line and uncomment the next 2 so save the graph to the server
    # experiment.log_figure(figure=plt)
    # plt.clf()

    print(confusion_matrix(saveInds, real))
