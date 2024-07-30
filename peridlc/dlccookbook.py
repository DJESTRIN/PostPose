import pandas as pd
import numpy as np
import tqdm
from scipy.interpolate import interp1d
import pickle
import os
import ipdb

class ingestion():
    """ Breaks DLC output csv files into common components for analyses """
    def __init__(self,csv_file,threshold=0.9,framerate=1):
        self.raw_data = pd.read_csv(csv_file) # Read file into df
        self.df = self.raw_data.iloc[1:,1:] #cut off unessesary data
        self.data = self.df.to_numpy() #convert to numpy array
        self.data = self.data.astype(float) #numpy array MUST be floats, not string/objs
        self.threshold=threshold
        self.framerate=framerate #frames per second

    def __call__(self):
        self.get_probabilities()
        self.interpolate()

    def get_probabilities(self):
        """ Break up data into x, y, & p. 
        Replace low probability events with nan """
        self.x,self.y,self.p=self.data[0::3],self.data[1::3],self.data[2::3] # Break up data into components

        # Replace low p with nan
        xup,yup=[],[]
        for xs,ys,ps in zip(self.x.T,self.y.T,self.p.T): #Loop over each column

            xsoh,ysoh=[],[] #place holder list for updated data
            for x,y,p in tqdm.tqdm(zip(xs,ys,ps),total=len(xs)): #Loop over each point, tqdm shows progress bar
                if p<self.treshold:
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
        super.__call__(self) # inherit call method above
        
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
        self.x_av = np.nanmean(self.x,axis=0) #nan mean ignores nan
        self.y_av = np.nanmean(self.y,axis=0)

        # Get metrics for average coordinates
        self.av_distance,self.av_speed,self.av_acc_mag = self.get_metrics(self.x_av,self.y_av)
    
    def get_metrics(self,xs,ys):
        """ Calculates the distance, speed and acceleration magnitute for any input coordinate data """
        distance=[] # Get distance
        for x1,x2,y1,y2 in zip(xs[:-1],xs[1:],ys[:-1],ys[1:]):
            distance.append(self.distance(x1,x2,y1,y2))
        distance=np.asarray(distance)

        speed=[]
        for d1,d2 in zip(distance[:-1],distance[1:]):
            self.speed(d1,d2)
        speed=np.asarray(speed)
        
        acc_mag=[]
        for s1,s2 in zip(speed[:-1],speed[1:]):
            self.acceleration_mag(s1,s2)
        acc_mag=np.asarray(acc_mag)
        return distance, speed, acc_mag

    def distance(self,x1,x2,y1,y2):
        """ Returns distance for coordinates """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def speed(self,d1,d2):
        """ Returns speed for distances """
        return (d2-d1)*self.framerate ### Is this right? change in distance / change in time

    def acceleration_mag(self,s1,s2):
        """ Returns acceleration magnitute for speeds """
        return (s2-s1)*self.framerate ### Is this right? Change in speed/ change in time
    
    @classmethod
    def load(cls,filename):
        """Load an instance from a pickle file."""
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    def save(self,filename):
        """Save the instance to a file using pickle."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    
class graphics():
    def __init__(self,digested_obj,output_directory,video_file=[]):
        # Did user include a video for us to use for plotting?
        if video_file:
            self.video_file=video_file
            self.attached_video=True
        else:
            self.attached_video=True

    def __call__(self):
        # Need to code these in later
        self.plot_trajectory()
        self.plot_distance()
        self.plot_speed()
        self.plot_acceleration()

def file_finder(directory):
    """ find all files of interest (csv, video, etc) in 
    current directory and put them in organized list """
    return

def main(inputfile,outputfile):
    """ Main set of steps for current analysis. """
    #Insert file finder function???

    # Determine if csv file has been analyzed before, load in data if true
    if os.path.isfile(outputfile):
        obj_oh = digestion.load(outputfile)
    else:
        obj_oh = digestion(inputfile)
        obj_oh()
        obj_oh.save(outputfile)
    
    # Generate graphics for current obj ... add in a loading feature later. 
    graph_obj = graphics(digestiondata=obj_oh)

if __name__=='__main__':
    fileoh = ''
    main(inputfile=fileoh,outputfile=r'\insert\some\directory\myobj.pkl')

