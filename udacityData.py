import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import torch
import glob
from collections import defaultdict
import cv2
from scipy.signal import convolve2d


trans={'horiz':transforms.RandomHorizontalFlip(p=1),
        'color':transforms.ColorJitter(.5),
        'rotate':None,
       'translate':None,
       'optical':None,
       'plain':None,
       'none':None}

imgTrans=transforms.ToPILImage()
tensTrans=transforms.ToTensor()
padding=transforms.Pad(100,padding_mode='symmetric')
crop=transforms.CenterCrop((480,640))

class UdacityData(Dataset):
    def __init__(self,transform,mode='train',amount=10, angNum=1):
        self.transform = transform
        self.mode=mode
        self.amount=amount
        self.angNum=angNum
        if self.mode=='train':
            self.path=None #ToDo: put the path to the udacity training images here
            self.data=pd.read_csv(self.path+'steering.csv')
        else:
            self.path=None #ToDo: put the path to the udacity testing images here
            self.data = pd.read_csv(self.path + 'final_example.csv',names=['timestamp','angle'])
            self.data = self.data[1:]


        # self.data=self.data[abs(self.data['steering_angle'])>.15]
        # self.data=self.data.sort_values('timestamp')
        times=self.data['timestamp'].astype('float').values
        angs=self.data['angle'].astype('float').values
        images=glob.glob(self.path+'center/*')
        imStamps=[int(x.split('/')[-1].split('.')[0]) for x in images]

        self.angles = defaultdict(list)
        for i,t in enumerate(times):
            if abs(angs[i])>=0:
                self.angles[t//(int(1e9)/20)].append(angs[i])
        self.images=defaultdict(list)
        for s in imStamps:
            self.images[s//(int(1e9)/20)].append(s)

        a_set = set(self.angles.keys())
        i_set = set(self.images.keys())
        self.validInds=list(a_set & i_set)


    def __len__(self):
        return len(self.validInds)-self.amount-self.angNum

    def normIm(self,im):
        A = [[np.min(im), 1], [np.max(im), 1]]
        B = [[0], [255]]
        q = np.linalg.solve(A, B)
        return im * q[0] + q[1]

    def __getitem__(self, item):
        # import pdb;
        # pdb.set_trace()
        if self.transform=='debug':
            imageInd = random.choice(self.images[self.validInds[item]])
            image=cv2.imread(self.path+'center/'+str(imageInd)+'.jpg')
            angle = np.mean(self.angles[self.validInds[item]])
            return self.validInds[item],image,angle

        images = []
        for i in range(self.amount):
            imageInd = random.choice(self.images[self.validInds[item + i]])
            images.append(torch.tensor(cv2.imread(self.path+'center/'+str(imageInd)+'.jpg')))

        transform = trans[self.transform]
        self.shiftAng=random.randrange(-25,25,1)
        self.transShift=random.randrange(-100,100,1)
        for i in range(len(images)):
            if transform:
                images[i]=tensTrans(transform(imgTrans(np.transpose(images[i],(-1, -3, -2)))))
                images[i] = -1 + 2 * images[i]
            elif self.transform=='optical':
                g2 = cv2.cvtColor(images[i].numpy(), cv2.COLOR_BGR2GRAY)
                ix = convolve2d(g2, [[1, 0, -1]], boundary='symm', mode='same')
                iy = convolve2d(g2, [[1], [0], [-1]], boundary='symm', mode='same')
                ix = self.normIm(ix)
                iy = self.normIm(iy)
                images[i]=torch.FloatTensor(-1+2*tensTrans(imgTrans(np.transpose(images[i],(-1, -3, -2)))))
                images.append(torch.FloatTensor([ix,ix,ix]))
                images.append(torch.FloatTensor([iy,iy,iy]))
            elif self.transform=='plain':
                images[i]=images[i]
            else:
                images[i]=tensTrans(imgTrans(np.transpose(images[i],(-1, -3, -2))))
                images[i]=-1+2*images[i]

        # returns the angle sequence
        if self.angNum>1:
            angle=[]
            for i in range(self.angNum):
                if self.transform == 'horiz':
                    angle.append(-np.mean(self.angles[self.validInds[item + self.amount - 1 + i]]))
                else:
                    angle.append(np.mean(self.angles[self.validInds[item + self.amount - 1 + i]]))
        else:
            if self.transform == 'horiz':
                angle=-np.mean(self.angles[self.validInds[item + self.amount - 1]])
            else:
                angle=np.mean(self.angles[self.validInds[item + self.amount - 1]])

        images=torch.stack(images)
        return [images,angle]

