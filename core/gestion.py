#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: gestion.py
Description: 
Author: David Estrin, Kenneth Johnson
Version: 1.0
Date: 08-29-2024
"""
# Import libraries and dependencies
import pandas as pd
import numpy as np
import tqdm
from scipy.interpolate import interp1d
import pickle
import ipdb

class ingestion():
    """ Breaks DLC output csv files into common components for analyses """
    def __init__(self,csv_file,drop_directory,threshold=0.9,framerate=1):
        self.raw_data = pd.read_csv(csv_file) # Read file into df
        self.df = self.raw_data.iloc[2:,1:] #cut off unessesary data
        self.data = self.df.to_numpy() #convert to numpy array
        self.data = self.data.astype(float) #numpy array MUST be floats, not string/objs
        self.threshold=threshold
        self.framerate=framerate #frames per second
        self.drop_directory=drop_directory

    def __call__(self):
        self.get_probabilities()
        self.interpolate()

    def get_probabilities(self):
        """ Break up data into x, y, & p. 
        Replace low probability events with nan """
        self.x,self.y,self.p=self.data[:,0::3],self.data[:,1::3],self.data[:,2::3] # Break up data into components

        # Replace low p with nan
        xup,yup=[],[]
        for xs,ys,ps in zip(self.x.T,self.y.T,self.p.T): #Loop over each column

            xsoh,ysoh=[],[] #place holder list for updated data
            for x,y,p in tqdm.tqdm(zip(xs,ys,ps),total=len(xs)): #Loop over each point, tqdm shows progress bar
                if p<self.threshold:
                    x=np.nan
                    y=np.nan
                xsoh.append(x)
                ysoh.append(y)
            xsoh,ysoh=np.asarray(xsoh),np.asarray(ysoh) #Convert list back to numpy
            xup.append(xsoh)
            yup.append(ysoh)
        
        xup,yup=np.asarray(xup),np.asarray(yup) #Convert list back to numpy array

        #Replace original data with updates regarding low probability times
        self.x,self.y=xup.T,yup.T
        
    def interpolate(self):
        """ Interpolate np.nan X and Y coordinates for each body part  """
        # Interpolate the X coordinates 
        x_interpolated=[]
        for bodypart in self.x.T: #loop over body parts
            if np.all(np.isnan(bodypart)):
                x_interpolated.append(bodypart)
                continue
            x_real=np.arange(len(bodypart))[~np.isnan(bodypart)]
            y_real=bodypart[~np.isnan(bodypart)]
            infunc = interp1d(x_real,y_real,kind='linear',fill_value='extrapolate')
            bd_int = bodypart.copy()
            bd_int[np.isnan(bodypart)] = infunc(np.arange(len(bodypart))[np.isnan(bodypart)])
            x_interpolated.append(bd_int)
        self.x=np.asarray(x_interpolated).T

        # Interpolate the Y coordinates
        y_interpolated=[]
        for bodypart in self.y.T: #loop over body parts
            if np.all(np.isnan(bodypart)):
                y_interpolated.append(bodypart)
                continue
            x_real=np.arange(len(bodypart))[~np.isnan(bodypart)]
            y_real=bodypart[~np.isnan(bodypart)]
            infunc = interp1d(x_real,y_real,kind='linear',fill_value='extrapolate')
            bd_int = bodypart.copy()
            bd_int[np.isnan(bodypart)] = infunc(np.arange(len(bodypart))[np.isnan(bodypart)])
            y_interpolated.append(bd_int)
        self.y=np.asarray(y_interpolated).T       

class digestion(ingestion):
    """ Performs essential analyses that are common for all behaviors """
    def __call__(self):
        """ Protocol for getting important data such as distances, speeds, accerlation magnitute for all body parts and average coordinates.
        inputs:
        self -- contains all necessary attributes

        outputs:
        self.bp_distances -- instantaneous distance calculated for each body part
        self.bp_speeds -- instantaneous speed calculated for each body part
        self.bp_acc_mags -- instantaneous acceleration magnitute calculated for each body part

        self.av_distance -- instantaneous distance calculated for average coordinates for all body parts
        self.av_speed -- instantaneous speed calculated for average coordinates for all body parts
        self.av_acc_mag -- instantaneous acceleration magnitute calculated for average coordinates for all body parts
        """
        super().__call__() # inherit call method above
        
        # Get metrics for each individual body part
        bp_distances,bp_speeds,bp_acc_mag=[],[],[]
        for body_part_xs, body_part_ys in zip(self.x.T,self.y.T):
            d_oh,s_oh,a_oh = self.get_metrics(body_part_xs,body_part_ys)
            bp_distances.append(d_oh)
            bp_speeds.append(s_oh)
            bp_acc_mag.append(a_oh)

        #Convert lists back to numpy arrays
        self.bp_distances = np.asarray(bp_distances)
        self.bp_speeds = np.asarray(bp_speeds)
        self.bp_acc_mags = np.asarray(bp_acc_mag)

        # Calculate average coordinates for all body parts
        self.x_av = np.nanmean(self.x,axis=1) #nan mean ignores nan
        self.y_av = np.nanmean(self.y,axis=1)

        # Get metrics for average coordinates
        self.av_distance,self.av_speed,self.av_acc_mag = self.get_metrics(self.x_av,self.y_av)
    
    def get_metrics(self,xs,ys):
        """ Calculates the distance, speed and acceleration magnitute for any input coordinate data """
        distance=[] # Get distance
        for x1,x2,y1,y2 in zip(xs[:-1],xs[1:],ys[:-1],ys[1:]):
            distance.append(self.distance(x1,x2,y1,y2))
        distance=np.asarray(distance)

        speed=[]
        for d1 in zip(distance[:-1]):
            speed.append(self.speed(d1,d2))
        speed=np.asarray(speed)
        
        acc_mag=[]
        for s1 in zip(speed[:-1]):
            acc_mag.append(self.acceleration_mag(s1))
        acc_mag=np.asarray(acc_mag)
        return distance, speed, acc_mag

    def distance(self,x1,x2,y1,y2):
        """ Returns distance for coordinates """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def speed(self,d1):
        """ Returns speed for distances """
        self.frametime=1/self.framerate
        return (d1)/self.frametime

    def acceleration_mag(self,s1):
        """ Returns acceleration magnitute for speeds """
        return (s1)/self.frametime 
    
    @classmethod
    def load(cls,filename):
        """Load an instance from a pickle file."""
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    def save(self,filename):
        """Save the instance to a file using pickle."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def get_stat(self,dependent_var,stat='mean'):
        """ A method for quickly calculating common stats for data 
        Inputs:
        self -- contains all necessary attributes
        dependent_var -- a N x 2 dimensional numpy array containing the start and end of timestamps for a given time of interest. Must contain 
            at least 3 rows of data. Otherwise, will throw error
        stat -- a string from the following list: ('mean','min','max','auc'). 'mean' will calculate the average amplitude of the dependent 
            variable across timestamps. 'min' and 'max' will calculate the minimumn and maximumn amplitudes of the dependent varialbe
            across timestamps, respectively. 'auc' will calculate the average area under the curve for the dependent variable during the give 
            timestamps. Default value is 'mean'.

        Outputs:
        Value -- The final value given the input data and timestamps. 
        StandardError -- A standard error is output in addition to the mean and auc. This is primarly meant for plotting
            purposes. 
        """
        if stat=='mean':
            Value = np.nanmean(dependent_var,axis=1) # Calculate the mean by row
            StandardError = np.nanstd(Value)/np.sqrt(Value.size) # Calculate the SE of the rows
            Value = np.nanmean(Value) # Calculate the overall mean        
        elif stat=='max':
            Value = np.nanmax(dependent_var,axis=1) # Calculate the mean by row
            StandardError = np.nanstd(Value)/np.sqrt(Value.size) # Calculate the SE of the rows
            Value = np.nanmean(Value) # Calculate the overall mean 
        elif stat=='min':
            Value = np.nanmin(dependent_var,axis=1) # Calculate the mean by row
            StandardError = np.nanstd(Value)/np.sqrt(Value.size) # Calculate the SE of the rows
            Value = np.nanmean(Value) # Calculate the overall mean 
        elif stat=='auc':
            try:
                Value = np.trapz(dependent_var,axis=1) # Calculate the mean by row
                StandardError = np.nanstd(Value)/np.sqrt(Value.size) # Calculate the SE of the rows
                Value = np.nanmean(Value) # Calculate the overall mean
            except:
                raise TypeError("When calculating the average AUC, an error occured with NaNs. Must modify code.")
        
        return Value, StandardError

    def get_stat_timeseries(self,timestamps,data,stat='mean'):
        """ 
        A method for quickly calculating common stats for timeseries data. 

        Inputs:
        self -- contains all necessary attributes
        timestamps -- a N x 2 dimensional numpy array containing the start and end of timestamps for a given time of interest. Must contain 
            at least 3 rows of data. Otherwise, will throw error
        data -- a N x 2 dimensional numpy array containing data for the time (column 0) and dependent variable (column 1). The time is used
            in conjunction with the timestamps above to determine where the dependent variable should be parsed. 
        stat -- a string from the following list: ('mean','min','max','auc'). 'mean' will calculate the average amplitude of the dependent 
            variable across timestamps. 'min' and 'max' will calculate the minimumn and maximumn amplitudes of the dependent varialbe
            across timestamps, respectively. 'auc' will calculate the average area under the curve for the dependent variable during the give 
            timestamps. Default value is 'mean'.

        Outputs:
        Value -- The final value given the input data and timestamps. 
        StandardError -- A standard error is output in addition to the mean and auc. This is primarly meant for plotting
            purposes. 
        """
        # Determine if timestamp and data are numpy arrays
        if not isinstance(timestamps,np.ndarray):
            raise TypeError("timestamps must be a numpy array.")
        if not isinstance(data,np.ndarray):
            raise TypeError("data must be a numpy array.")
        
        # Determine if timestamps and data arrays are the correct shape
        rows, cols = timestamps.shape
        if rows < 3:
            raise TypeError("timestamps must have at least 3 rows of data")
        if cols!=2:
            raise TypeError("timestamps must have 2 columns (start times,stop times)")
        
        rows, cols = data.shape
        if rows < 3:
            raise TypeError("data must have at least 3 rows of data")
        if cols!=2:
            raise TypeError("data must have 2 columns (time, dependent variable of interest)")
        
        # Loop over timestamps, parse data by time and collect dependent variable into new array. 
        dependent_var=[]
        for (start,stop) in timestamps:
            dependent_var.append(data[np.where(data[:,0]>start and data[:,0]<stop)[0],1])
        dependent_var=np.asarray(dependent_var) # Convert list back to numpy

        Value, StandardError = self.get_stat(dependent_var,stat)
        return Value, StandardError

if __name__=='__main__':
    objoh = digestion(r'C:\Users\listo\PostPose\test_data\24-7-2_C4478776_M2DLC_resnet50_open_fieldMay10shuffle1_200000.csv',
                      r'C:\Users\listo\PostPose\test_data')
    objoh()
    ipdb.set_trace()