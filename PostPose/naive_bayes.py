#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:05:38 2023

@author: dje4001
"""

""" Features
    deeplabcut postition qtip, deeplabcut probability, pixel_value
    
    answer, 0 (no) or 1 (yes)
    """
import sys,glob
sys.path.append('/home/dje4001/post_dlc/')
from parse import hdf_to_tall
import cv2
import numpy as np
import ipdb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB
import torch.nn as nn
import torch as t 
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class generate_training_data(hdf_to_tall):
    def get_pixel_values(self):
        # Get corresponding video file
        video,_=self.file_string.split('DLC')
        video=[video+'.mp4']
        _,_,_,_,_,_,_,self.video_filename=video[0].split('/')
        # Get pixel values
        qbase_a=[]
        qmid_a=[]
        qtip_a=[]
        qbase_p=[]
        qmid_p=[]
        qtip_p=[]
        frames=[]
        cap=cv2.VideoCapture(video[0])
        qtip_ps=self.ps[self.ps.columns[25:28,]]
        for i,(x,y,ps) in enumerate(zip(self.qtip_xs.to_numpy(),self.qtip_ys.to_numpy(),qtip_ps.to_numpy())):
            if i%103==0: #There are too many frames so I am cutting down data with this
                cap.set(1,i)
                success=cap.grab()
                ret,image=cap.retrieve()
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=np.array(image)
                
                #Grab pixel values for each part of qtip
                try:
                    qbase=image[round(x[0]),round(y[0]),:]
                except:
                    qbase=np.array([np.nan,np.nan,np.nan])
                
                try:
                    qmid=image[round(x[1]),round(y[1]),:]
                except:
                    qmid=np.array([np.nan,np.nan,np.nan])
                    
                try:
                    qtip=image[round(x[2]),round(y[2]),:]
                except:
                    qtip=np.array([np.nan,np.nan,np.nan])
    
                qbase_a.append(qbase)
                qmid_a.append(qmid)
                qtip_a.append(qtip)
                qbase_p.append(ps[0])
                qmid_p.append(ps[1])
                qtip_p.append(ps[2])
                frames.append(i)
        
        qbase_a=np.array(qbase_a)
        qmid_a=np.array(qmid_a)
        qbase_p=np.array(qbase_p)
        qtip_a=np.array(qtip_a)
        qmid_p=np.array(qmid_p)
        qtip_p=np.array(qtip_p)
        frames=np.array(frames)
        
        #horizontal concatenate 
        qbase_a=np.hstack((qbase_a,np.expand_dims(qbase_p,axis=1)))
        qmid_a=np.hstack((qmid_a,np.expand_dims(qmid_p,axis=1)))
        qtip_a=np.hstack((qtip_a,np.expand_dims(qtip_p,axis=1)))
        self.qf_a=np.hstack((qbase_a,qmid_a,qtip_a,np.expand_dims(frames,axis=1)))
    
    def get_training_data(self,video_names,onsets,offsets):
        Y=[]
        if self.video_filename in video_names:
            onset,offset=round(onsets[video_names.index(self.video_filename)]*len(self.qtip_xs)),round(offsets[video_names.index(self.video_filename)]*len(self.qtip_xs))
            for feature in self.qf_a:
                if onset<=feature[-1] and feature[-1]<=offset:
                    Y.append(1)
                else:
                    Y.append(0)
                    
        else:
            for feature in range(len(self.qf_a)):
                    Y.append(0)
                        
        return self.qf_a, np.array(Y)


""" Custom neural network class (MLP) """
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(13, 50) 
        self.linear2 = nn.Linear(50, 5) 
        self.final = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, xs): #convert + flatten
        x1 = self.relu(self.linear1(xs))
        x2 = self.relu(self.linear2(x1))
        xf = self.final(x2)
        return xf
     
""" Script inputs """
training_directory='/athena/listonlab/store/dje4001/deeplabcut/qtiptraining/'
video_names=['Date_08_16_2022_15_10_46_9_3976688_10800_A_1_male_40_0g_08_16_2022_FosCreERxAi9_NA_experimentalday4.mp4',
             'Date_08_16_2022_15_10_44_7_3976646_10800_A_2_male_40_4g_08_16_2022_FosCreERxAi9_NA_experimentalday5.mp4',
             'Date_08_16_2022_15_10_44_7_3976646_10800_A_2_male_40_4g_08_16_2022_FosCreERxAi9_NA_experimentalday4.mp4']
onsets=[0.71277,0,0.6907] #percent of video
offsets=[1,0.6157,1] # percent of the video
            
""" Generate training dataset """
h5_files=glob.glob(training_directory+'*.h5')
X_all=[]
y_all=[]
for h5oh in h5_files:
    print(h5oh)
    data=generate_training_data(h5oh)
    data.get_pixel_values()
    X,y=data.get_training_data(video_names,onsets,offsets)
    X_all.append(X)
    y_all.append(y)

""" Convert training data to torch data set """
# Correct training data format
X=np.array(X_all)
y=np.array(y_all)
w,l,f=X.shape
X=np.reshape(X,(f,l*w))
y=np.reshape(y,(l*w))
X=X.T

#Cut down zero inflation:
zeros=np.array(np.where(y==0))
zeros=zeros.flatten()
ones=np.array(np.where(y==1))
ones=ones.flatten()
indexes=np.random.choice(zeros,ones.shape[0],replace=False)
X_new=np.vstack((X[indexes],X[ones]))
y_new=np.hstack((y[indexes],y[ones]))

#Split data
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.15, random_state=3)

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)

training_dataset = t.utils.data.TensorDataset(t.tensor(X_train_imp), t.tensor(y_train))
train_loader = t.utils.data.DataLoader(training_dataset,shuffle=True,batch_size=64)

""" Train neural network """
def training(network,losstype,optimizer,epoch,train_loader):
    lossf=[]
    f1f=[]
    for epoch in range(epoch):
        network.train()
        lossoh=[]
        f1oh=[]
        for d in train_loader:
            x, y = d
            x=x.to(dtype=t.float32)
            losstype.zero_grad()
            y_pred= net(x)
            loss = cross_el(y_pred, y)
            lossoh.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            yp=t.argmax(y_pred,axis=1)
            f1oh.append(f1_score(y,yp))
            
        lossf.append(lossoh)
        f1f.append(f1oh)
    return lossf,f1f

net = Net()
optimizer = t.optim.Adam(net.parameters(), lr=0.001) #e-1
cross_el = nn.CrossEntropyLoss()
losses,f1f=training(net,cross_el,optimizer,1000,train_loader)

f1f=np.array(f1f)
f1f=np.mean(f1f,axis=1)
losses=np.array(losses)
losses=np.mean(losses,axis=1)
plt.figure()
plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("loss")

plt.figure()
plt.plot(f1f)
plt.xlabel("epochs")
plt.ylabel("f1")

clf=RandomForestClassifier(max_depth=100,random_state=1)
clf.fit(X_train_imp,y_train)
predic=clf.predict(X_test_imp)






