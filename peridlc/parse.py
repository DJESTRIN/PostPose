#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parseDLC
Code for taking DLC output and getting basic behavioral data.


Things to check:
    (1) Are the h5 outputs still the same locations
    (2) Add in is q tip present or not. (before, during, after)
    (3) Are dataframes combining correctly?
    (4) Parse by TMT vs Water. 
    
    """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import cv2
import seaborn as sns
import math
from scipy import integrate
from scipy.stats import zscore

class hdf_to_tall(object):
    def __init__(self,file_string):
        #file string contains fullpath to data
        self.file_string=file_string
        self.data=pd.read_hdf(file_string)
        self.forward()
        
    def forward(self):
        self.strip_name()
        self.get_start_time()
        self.divide_dataframe()
        self.measurement_key()
        
        #Distance between mouse and qtip base
        self.mouse_d=self.distance(self.mouse_avx,np.tile(self.qtip_avx[0],(self.mouse_avx.shape[0],)).T,
                                   self.mouse_avy,np.tile(self.qtip_avy[0],(self.mouse_avy.shape[0],)).T)/self.key
        
        #Velocity (cm/s) of mouse
        mdsx=self.mouse_avx[::32] #downsampled to 1 frame per second
        mdsy=self.mouse_avy[::32]
        self.mouse_v=self.distance(mdsx[1:],mdsy[1:],mdsx[0:-1],mdsy[0:-1])/self.key
        
        #Plot figures for internal use
        timerds=self.time[::32]
        self.timerds=timerds[0:-1]

        #Qtip bool
        self.qtip_bool()
        
        #Generate and save tall datasets
        self.BuildTall()
        
    def strip_name(self):
        self.basename=os.path.basename(self.file_string)
        self.basename,_=self.basename.split('DLC')
        _,self.month,self.day,self.year,self.hour,self.minute,self.second,self.box,self.cage,_,_,self.subjectid,self.sex,self.weight,_,_,_,_,self.strain,self.condition,self.sessionname=self.basename.split('_')
        return
    
    def get_start_time(self):
        """Video is divided into 10 sections, find the section and then alter the time to the 
        correct video time"""
        
        #Poorly coded but will probably fix later
        if '10' in self.sessionname:
            start=9
        elif '9' in self.sessionname:
            start=8
        elif '8' in self.sessionname:
            start=7
        elif '7' in self.sessionname:
            start=6
        elif '6' in self.sessionname:
            start=5
        elif '5' in self.sessionname:
            start=4
        elif '4' in self.sessionname:
            start=3
        elif '3' in self.sessionname:
            start=2
        elif '2' in self.sessionname:
            start=1
        else:
            start=0
        
        #Create a Time column in dataframe
        starting_time=18*60*start
        self.time=np.linspace(0,self.data.shape[0]*(1/32),
                              num=self.data.shape[0])+starting_time
        self.data['Time']=self.time
        return
    
    def divide_dataframe(self):
        """Seperate the data frame into Mouse, Qtip and Corners dataframes
        Hard coded :( 
            0->74 === Mouse
            75->83 === Qtip
            84->95 === corners
        """
        # Remove data with low probability <0.9
        xs=self.data[self.data.columns[0::3]]
        ys=self.data[self.data.columns[1::3]]
        self.ps=self.data[self.data.columns[2::3]]
        
        # Get mouse X and y coordinates
        mouse_xs=xs[xs.columns[0:24,]]
        mouse_ys=ys[ys.columns[0:24,]]
        mouse_xs_face=mouse_xs[mouse_xs.columns[0:5,]]
        mouse_ys_face=mouse_ys[mouse_ys.columns[0:5,]]
        
        self.mouse_face_x=mouse_xs_face.mean(axis=1)
        self.mouse_face_y=mouse_ys_face.mean(axis=1)
        self.mouse_avx=mouse_xs.median(axis=1)
        self.mouse_avy=mouse_ys.median(axis=1)
        
        #Get qtip coordinates
        self.qtip_xs=xs[xs.columns[25:28,]]
        self.qtip_ys=ys[ys.columns[25:28,]]
        self.qtip_ps=self.ps[self.ps.columns[27:28,]]
        self.qtip_avx=self.qtip_xs.median()
        self.qtip_avy=self.qtip_ys.median()
        
        #Get Corner coordinates
        corners_xs=xs[xs.columns[28:32,]]
        corners_ys=ys[ys.columns[28:32,]]
        self.cornersx=corners_xs.median()
        self.cornersy=corners_ys.median()
    
    def graphs(self):
        #get video name
        file,_=self.file_string.split('DLC')
        _,title=file.split('Date_')
        file=file+'.mp4'
        cap=cv2.VideoCapture(file)
        cap.set(1,600)
        success=cap.grab()
        ret,image=cap.retrieve()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=np.array(image)
        figs,axs=plt.subplots(1,4)
        axs[0].imshow(image,aspect='auto')
        axs[0].plot(self.mouse_avx,self.mouse_avy,alpha=0.6,c="green")
        axs[0].scatter(self.qtip_avx[0],self.qtip_avy[0],s=20,c="red")
        axs[0].scatter(self.cornersx,self.cornersy,s=10,c="blue")
        axs[0].set_xlim([min(self.cornersx)-10,max(self.cornersx)+10])
        axs[0].set_ylim([min(self.cornersy)-10,max(self.cornersy)+10])
        axs[0].axis("off")
        try:
            axs[1].hist2d(self.mouse_avx,self.mouse_avy, bins=100,norm=matplotlib.colors.LogNorm(),range=[[min(self.cornersx)-10,max(self.cornersx)+10],[min(self.cornersy)-10,max(self.cornersy)+10]])
            axs[1].scatter(self.qtip_avx[0],self.qtip_avy[0],s=20,c="red")
            axs[1].scatter(self.cornersx,self.cornersy,s=10,c="blue")
            axs[1].axis("off") 
        except:
            print(self.file_string)
            
        axs[2].plot(self.time,self.mouse_d)
        axs[2].set_xlabel("Time(s)")
        axs[2].set_ylabel("Distance from Q Tip (cm)")
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[3].plot(self.timerds,self.mouse_v.T)
        axs[3].set_xlabel("Time(s)")
        axs[3].set_ylabel("Velocity (cm/s)")
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['right'].set_visible(False)
        #axs[0,2].scatter(self.time,self.qtip_bool_df,c=self.qtip_bool_df)
        #axs[0,2].set_ylim([-0.25,1.25])
        #axs[0,2].text(self.time.mean(),0.1,"No Qtip")
        #axs[0,2].text(self.time.mean(),0.9,"Qtip")
        #axs[0,2].axis("off")
        plt.tight_layout()
        plt.show()
        #plt.title(title)
        return
    
    def contact_qtip(self):
        """Bool value indicating where mouse nose in within threshold of qtip """
        """To get duration, sum over and then multiple by sample rate """
        self.contact_qtip_bool=np.zeros(self.mouse_face_x.shape)
        counter=0
        for x,y in zip(self.mouse_face_x[1:],self.mouse_face_y[1:]):
            distance=math.dist([x,y],[self.qtip_avx[0],self.qtip_avy[0]])
            if distance <50: #50 pixels is my arbitray threshold
                self.contact_qtip_bool[counter,]=1
            counter+=1
            
        
    def far_corner_duration(self):
        """ Get duration mice are in the FAR corners of their box """
        #find "far" corner
        fx,fy=self.qtip_avx[0],self.qtip_avy[0]
        sfx,sfy=self.qtip_avx[0],self.qtip_avy[0]
        for x,y in zip(self.cornersx,self.cornersy):
            if math.dist([self.qtip_avx[0],self.qtip_avy[0]],[sfx,sfy]) < math.dist([self.qtip_avx[0],self.qtip_avy[0]],[x,y]):
                if math.dist([self.qtip_avx[0],self.qtip_avy[0]],[fx,fy]) < math.dist([self.qtip_avx[0],self.qtip_avy[0]],[x,y]):
                    fx=x
                    fy=y
                else:        
                    sfx=x
                    sfy=y
        
        #For graphing
        self.far_cornersx=[fx,sfx]
        self.far_cornersy=[fy,sfy]
        
        self.far_corner_bool=np.zeros(self.mouse_avx.shape)
        counter=0
        for x,y in zip(self.mouse_avx[1:],self.mouse_avy[1:]):
            distance1=math.dist([x,y],[fx,fy])
            distance2=math.dist([x,y],[sfx,sfy])
            if distance1<75 or distance2<75: #70 pixels is my arbitray threshold
                self.far_corner_bool[counter,]=1
            counter+=1
        return
        
    def distance_traveled(self):
        """ Calculate cumulative distance traveled """
        return sum(self.mouse_v)
        
    def measurement_key(self):
        self.corner_distance=self.distance(self.cornersx[0],self.cornersy[0],self.cornersx[1],self.cornersy[1])
        self.key=self.corner_distance/13.335 # The distance in cm between corner 1 and 2
        return
        
    def qtip_bool(self):
        #Determine if the qtip is present... 
        #probs should be the ps dataframe or a variation of this
        self.qtip_bool_df = np.zeros(shape=[self.qtip_ps.shape[0], 1])
        self.qtip_bool_df[self.qtip_ps[:]>0.95]=1
        self.qtip_bool_df=pd.DataFrame(self.qtip_bool_df)
        qtipds=self.qtip_bool_df[::32]
        self.qtip_bool_dfds=qtipds[0:-1]
        return 
        
        
    def distance(self,x1,y1,x2,y2):
        x1,y1,x2,y2=np.array(x1),np.array(y1),np.array(x2),np.array(y2)
        return (np.sqrt((x1-x2)**2+(y1-y2)**2))
                
    def BuildTall(self):
        #Create dataframe based on distance or velocity
        self.info=np.array([self.cage,self.subjectid,self.box,self.sex,self.weight,self.strain,self.condition,self.sessionname])
        
        #Build tall for distance data
        self.infod=np.tile(self.info,(self.mouse_d.shape[0],1))
        self.repeated_info=pd.DataFrame({'cage':self.infod[:,0], 'subjectid':self.infod[:,1],
                      'box': self.infod[:,2], 'sex':self.infod[:,3], 'weight':self.infod[:,4],
                      'strain':self.infod[:,5], 'condition':self.infod[:,6], 'sessionname':self.infod[:,7]})
        
        self.distance_tall=pd.concat([self.repeated_info,
                                      self.qtip_bool_df[:],
                                      pd.DataFrame({'time':self.time}), 
                                      pd.DataFrame({'distance':self.mouse_d})],axis=1)
        
        #Build tall for velocity data
        self.infov=np.tile(self.info,(self.mouse_v.shape[0],1))
        self.repeated_info=pd.DataFrame({'cage':self.infov[:,0], 'subjectid':self.infov[:,1],
                      'box': self.infov[:,2], 'sex':self.infov[:,3], 'weight':self.infov[:,4],
                      'strain':self.infov[:,5], 'condition':self.infov[:,6], 'sessionname':self.infov[:,7]})
        
        self.velocity_tall=pd.concat([self.repeated_info,
                                      self.qtip_bool_dfds[:],
                                      pd.DataFrame({'time':self.timerds}), 
                                      pd.DataFrame({'velocity':self.mouse_v})],axis=1)
        
        filename=os.path.dirname(self.file_string)+"/"+str(self.cage)+str(self.subjectid)+str(self.box)+str(self.strain)+str(self.sessionname)+"_distance.csv"
        self.distance_tall.to_csv(filename,index=False)
        filename=os.path.dirname(self.file_string)+"/"+str(self.cage)+str(self.subjectid)+str(self.box)+str(self.strain)+str(self.sessionname)+"_velocity.csv"
        self.velocity_tall.to_csv(filename,index=False)
        return


""" (4) Parse h5 files and output as csv files """
if __name__=="__main__":
    import glob
    import ipdb
    h5_files=glob.glob("/athena/listonlab/store/dje4001/deeplabcut/processed_video_drop/*.h5")
    for h5oh in h5_files:
        data=hdf_to_tall(h5oh)
        data.graphs()
        data.contact_qtip()
        data.far_corner_duration()
        #ipdb.set_trace()
      
    all_files=[i for i in glob.glob('*distance.csv')]
    combined_csv=pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv.to_csv("08_16_2022_distance.csv",index=False)    
      
    all_files=[i for i in glob.glob('*velocity.csv')]
    combined_csv=pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv.to_csv("08_16_2022_distance.csv",index=False)      

# """ When called, run for video file path """
# parser=argparse.ArgumentParser()
# parser.add_argument("--h5_dir",type=str,required=True) #input for video file

# if __name__=="__main__":
#     args=parser.parse_args()
#     data=hdf_to_tall(args.h5_dir)


""" 
Make tall matrix per file:
    Cage, Animal, Virus, Condition (TMT or water),Time (keeping in mind video number), 
    X coordinate, y coordinate, 
    Body Center Distance from base, Velocity, Presence of q tip (QTIP vs NOQTIP), Tip coordinates. 

    Calculate pixels to distance using Corners...
"""
    
    

