#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parseDLC
Code for taking DLC output and getting basic behavioral data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import ipdb

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
        self.plot(self.time,self.mouse_d)
        timerds=self.time[::32]
        timerds=timerds[0:-1]
        self.plot(timerds,self.mouse_v.T)
        
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
        self.mouse_avx=mouse_xs.mean(axis=1)
        self.mouse_avy=mouse_ys.mean(axis=1)
        
        #Get qtip coordinates
        self.qtip_xs=xs[xs.columns[25:26,]]
        self.qtip_ys=ys[ys.columns[25:28,]]
        self.qtip_ps=self.ps[self.ps.columns[27:28,]]
        self.qtip_avx=self.qtip_xs.mean()
        self.qtip_avy=self.qtip_ys.mean()
        
        #Get Corner coordinates
        corners_xs=xs[xs.columns[28:32,]]
        corners_ys=ys[ys.columns[28:32,]]
        self.cornersx=corners_xs.mean()
        self.cornersy=corners_ys.mean()
        
    def measurement_key(self):
        self.corner_distance=self.distance(self.cornersx[0],self.cornersy[0],self.cornersx[1],self.cornersy[1])
        self.key=self.corner_distance/13.335 # The distance in cm between corner 1 and 2
        return
        
    def qtip_bool(self):
        #Determine if the qtip is present... 
        #probs should be the ps dataframe or a variation of this
        self.qtip_bool_df = np.zeros(shape=[self.qtip_ps.shape[0], 1])
        self.qtip_bool_df[self.qtip_ps[:]>0.9]=1
        self.qtip_bool_df=pd.DataFrame(self.qtip_bool_df)
        return 
        
        
    def distance(self,x1,y1,x2,y2):
        x1,y1,x2,y2=np.array(x1),np.array(y1),np.array(x2),np.array(y2)
        return (np.sqrt((x2-x1)**2+(y2-y1)**2))
    
    def plot(self,x,y):
        plt.figure()
        plt.plot(x,y)
        return
                
    def BuildTall(self):
        #Create dataframe based on distance or velocity
        self.info=np.array([self.cage,self.subjectid,self.box,self.sex,self.weight,self.strain,self.condition,self.sessionname])
        
        #Build tall for distance data
        self.info=np.tile(self.info,(self.mouse_d.shape[0],1))
        self.repeated_info=pd.DataFrame({'cage':self.info[:,0], 'subjectid':self.info[:,1],
                      'box': self.info[:,2], 'sex':self.info[:,3], 'weight':self.info[:,4],
                      'strain':self.info[:,5], 'condition':self.info[:,6], 'sessionname':self.info[:,7]})
        
        self.distance_tall=pd.concat([self.repeated_info,
                                      pd.DataFrame({'time':self.time}), 
                                      pd.DataFrame({'distance':self.mouse_d})],axis=1)
        
        filename=os.path.dirname(self.file_string)+"/"+str(self.cage)+str(self.subjectid)+str(self.box)+str(self.strain)+str(self.sessionname)+".csv"
        print(filename)
        self.distance_tall.to_csv(filename,index=False)
        return



""" When called, run for video file path """
parser=argparse.ArgumentParser()
parser.add_argument("--video_dir",type=str,required=True) #input for video file

if __name__=="__main__":
    args=parser.parse_args()
    data=hdf_to_tall(args.video_dir)


""" 
Make tall matrix per file:
    Cage, Animal, Virus, Condition (TMT or water),Time (keeping in mind video number), 
    X coordinate, y coordinate, 
    Body Center Distance from base, Velocity, Presence of q tip (QTIP vs NOQTIP), Tip coordinates. 

    Calculate pixels to distance using Corners...
"""
    
    

