from sklearn.manifold import TSNE
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import glob
import cv2
import random

coords=[]
# records the coordinates of the mouse click
def onclick(event):
    global ix, iy
    ix,iy = event.xdata, event.ydata
    global coords
    coords=[(ix,iy)]
    if len(coords)>0:
        # print(coords)
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)

#finds the nearest value in the array of points to the location of the mouse click
def find_nearest(array,value):
    temp=np.abs(array - value)
    small=np.abs(array - value)
    small.sort()
    small=small[:10]
    diff=1e6
    for s in small:
        p=np.where(temp==s)
        d=np.sum(abs(data[p]-coords))
        if d<diff:
            diff=d
            point=p
    return point

path=None #ToDo: put the base path to the folder where the images, braking, angle, and throttle csv files are
brakes=pd.read_csv(path+'brake.csv').sort_values('timestamp')
angles=pd.read_csv(path+'steering.csv').sort_values('timestamp')
throttle=pd.read_csv(path+'throttle.csv').sort_values('timestamp')
images=glob.glob(path+'center/*')
imStamps=[int(x.split('/')[-1].split('.')[0]) for x in images]

#collate the angle, image, braking, and throttle data
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

a_set = set(AngData.keys())
b_set=set(BrakeData.keys())
t_set=set(ThrotData.keys())
validInds=list(a_set & b_set & t_set)

angCol=[]
braCol=[]
thrCol=[]
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



    angCol.append(atemp)
    aLabel.append(aVal/40)
    braCol.append(btemp)
    thrCol.append(ttemp)
    atemp.extend(btemp)
    atemp.extend(ttemp)
    fullData.append(atemp)
    i+=40


#create the TSNE embedding
tsneA=TSNE()
data=tsneA.fit_transform(np.array(fullData))


# start the plot, allow the user to click a point, and show the relevant image frames
while(True):
    coords=[]
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.scatter(data[:, 0], data[:, 1], c=aLabel, alpha=0.5)
    plt.title('Angle Data TSNE - 40 steps')
    plt.show()

    point1 = find_nearest(tsneA.embedding_[:,0], coords[0][0])
    point2 = find_nearest(tsneA.embedding_[:,1], coords[0][1])

    diff1=np.abs(data[point1]-coords)
    diff2=np.abs(data[point2]-coords)
    if np.sum(diff1)<=np.sum(diff2):
        point=point1[0][0]
    else:
        point=point2[0][0]


    plt.scatter(data[:, 0], data[:, 1], c=aLabel, alpha=0.5)
    plt.scatter(data[point][0], data[point][1], c='r', alpha=1)
    plt.title('Angle Data TSNE - 40 steps')
    plt.draw()
    plt.pause(1)
    plt.close()

    imageInd=int(validInds[point*40])
    for i in range(imageInd,imageInd+40):
        image=cv2.imread(path+'center/'+str(random.choice(ImgData[i]))+'.jpg')
        cv2.imshow('', image)
        key = cv2.waitKey(50)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
