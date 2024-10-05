#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: headrestricted.py
Description: For analysing head restricted tasks on the two-photon. Primarily will be used for analysing nose/ body movement during time-locked
    stimuli. May eventually include eye movement data. 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""
from gestion import digestion
from main import pipeline, delete_saved_objects
from graphics import graphics, experimental_field
import argparse
import numpy as np
import os, re
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import pandas as pd
import ipdb

# Take both front and side video and merge into single objects
# Run code without a experimental field shape
# Re calculate movement metrics .... 

class headrestricted_graphics(graphics):
    """ head restricted graphics ... uses graphics class to calculate the following details:
        (1) Calculate movement metrics (velocity, distance, acceleration) of body and body parts. 
        (2) Calculate movement metrics before and after each trial ...  
    """
    def __call__(self):
        super().__call__()
        self.calculate_percent_time()
        self.number_entries_innercircle = self.calculate_transitions_innercircle()
        #self.circle_metrics()

    def is_inside_circle(self,x_trajectory,y_trajectory, x_center, y_center, radius):
        distances_squared = (x_trajectory - x_center) ** 2 + (y_trajectory - y_center) ** 2
        return distances_squared <= radius ** 2

    def calculate_percent_time(self,x=None,y=None):
        if (x is None) or (y is None): 
            x=self.digested_obj.x_av ### NEED TO FIX THIS...
            y=self.digested_obj.y_av
        
        # Eliminate points outside of outter circle
        x_center,y_center,radius=self.arena_obj.shape_positions[0]
        boolout = self.is_inside_circle(x_trajectory=x,y_trajectory=y,x_center=x_center,y_center=y_center,radius=radius)
        x,y=x[boolout],y[boolout]

        # Determine which points are inside inner circle
        x_center,y_center,radius=self.arena_obj.shape_positions[1]
        boolout = self.is_inside_circle(x_trajectory=x,y_trajectory=y,x_center=x_center,y_center=y_center,radius=radius)
        x_inner,y_inner=x[boolout],y[boolout]
        x_outer,y_outer=x[~boolout],y[~boolout]
        
        self.percent_time_inner=len(x_inner)/(len(x_inner)+len(x_outer))
        self.percent_time_outer=len(x_outer)/(len(x_inner)+len(x_outer))
        self.inner_circle_boolean=boolout

    def calculate_transitions_innercircle(self):
        transitions = np.where((self.inner_circle_boolean[:-1] == False) & (self.inner_circle_boolean[1:] == True))[0]
        return len(transitions)
    
    def circle_metrics(self):
        """ circle metrics: re-calculate the distance, speed and acceleration_mag 
            inside the inner circle versus outer circle """
        # Calculate cumulative distance in inner vs outer circle
        distance_inner = self.digested_obj.av_distance[self.inner_circle_boolean[:-1]]
        distance_outer = self.digested_obj.av_distance[~self.inner_circle_boolean[:-1]]
        self.total_distance_inner=np.sum(distance_inner) # In pixels ==> NEED TO CONVERT TO CM
        self.total_distance_outer=np.sum(distance_outer)

        # Calculate average speed in inner vs outer circle
        speed_inner = self.digested_obj.av_speed[self.inner_circle_boolean[:-2]]
        speed_outer = self.digested_obj.av_speed[~self.inner_circle_boolean[:-2]]
        self.average_speed_inner=np.mean(speed_inner) # In pixels ==> NEED TO CONVERT TO CM
        self.average_speed_outer=np.mean(speed_outer)

        # Calculate average acceleration magnitude in inner vs outer circle
        acc_inner = self.digested_obj.av_acc_mag[self.inner_circle_boolean[:-3]]
        acc_outer = self.digested_obj.av_acc_mag[~self.inner_circle_boolean[:-3]]
        self.average_acc_inner=np.mean(acc_inner) # In pixels ==> NEED TO CONVERT TO CM
        self.average_acc_outer=np.mean(acc_outer)
        ipdb.set_trace()

class openfield_pipeline(pipeline):
    """ Updated pipeline class to accomodate openfield_graphics class """
    def __call__(self):
        """ Main set of steps for current analysis. """
        # Loop over csv files, making sure all csv files have been processed.
        self.digestion_objs = []
        self.arena_objs = []
        self.graphics_objs=[]

        for csvfile in self.csv_files:
            outputfile,_=csvfile.split('.cs')
            outputfile+='.pkl'

            # Determine if object was already created on a previous run.
            if os.path.isfile(outputfile):
                obj_oh = digestion.load(outputfile)
            else:
                obj_oh = digestion(csv_file=csvfile,framerate=60,cms_per_pixel=0.245,rolling_window=60)
                obj_oh()
                obj_oh.save(outputfile)
            
            # Keep all digestion objects inside of a single attribute
            self.digestion_objs.append(obj_oh)

            # Get corresponding video file
            video_file = self.match_csv_to_video(csvfile)

            # Create or load corresponding arena object
            field_file,_=re.split(r'\.avi|\.mp4', video_file)
            field_file+='experimental_field.pkl'
            if os.path.isfile(field_file):
                arena_objoh = experimental_field.load(field_file)
            else:
                arena_objoh = experimental_field(input_video=video_file,
                                           drop_directory=self.drop_directory,
                                           shape_positions=self.shape_positions[0],
                                           shapes=self.shapes[0])
                arena_objoh.save(field_file)
        
            # Keep all arena objects inside a single attribute
            self.arena_objs.append(arena_objoh)

            # Create or load graphics object
            graphics_file,_=re.split(r'\.avi|\.mp4', video_file)
            graphics_file+='graphics.pkl'
            if os.path.isfile(graphics_file):
                graph_obj = openfield_graphics.load(graphics_file)
            else:
                graph_obj = openfield_graphics(digested_obj=obj_oh,
                                    arena_obj=arena_objoh,
                                    drop_directory=self.drop_directory)
                graph_obj()
                graph_obj.save(field_file)
            
            # Keep all graphics objects inside a single attribute
            self.graphics_objs.append(graph_obj)

class openfield_statistics(openfield_pipeline):
    """ Generate statistics
    Description: This class is meant to pull all of the important data from each digestion object across groups 
        and capture statistics for each group. 
    """
    def build_tables(self,dependent_variables=["number_entries_innercircle"],independent_variables=["cage","mouse","day","group"]):
        """ Build tables -- this method takes dependent variables and generates an aggregated table of the dependent variable
            given the list of independent variables.
        """
        self.tables=[] # a list of all pandas tables
        
        for depend in dependent_variables: # Loop over dependent variables
            for i,digobjoh in enumerate(self.graphics_objs): # Loop over digestion objects and pull data
                if hasattr(digobjoh,depend):

                    missing_attributes = [attr for attr in independent_variables if not hasattr(digobjoh, attr)]
                    if missing_attributes:
                        print("missing attributes:", missing_attributes)
                        raise("Missing independent variables as attributes, must look at your spelling or if independent variables not generated")
                    else:
                        attribute_values = [getattr(digobjoh, attr, None) for attr in independent_variables]

                        # Append data to dataframe 
                        if i==0:
                            table_f=pd.DataFrame({})
                        else:
                            table_oh=[]
                            table_f+=table_oh
                else:
                    raise("Dependent variable is not an attribute")
                
            self.tables.append(table_f)

        # for digobjoh in self.graphics_objs:
        #     """ Pull Data + make table ...
        #     Cage, SubjectID, Day (0,7?, 14,30), Cohort (1,2,np.nan), Group (cort, control), Behavior (open_field),  
        #         number_entries_innercircle, percent time inner circle, percent time outer circle, distance_inner_circle, distance_outer_circle,
        #         speed_inner_circle, speed_outer_circle, acc_mag_inner_circle, acc_mag_outer_circle
        #     """
        #     ipdb.set_trace()
            
        #     # Example data
        #     exampledata = {
        #         'Subject': [1, 1, 2, 2, 3, 3],
        #         'Group': ['A', 'B', 'A', 'B', 'A', 'B'],
        #         'Session': ['day1', 'day1', 'day1', 'day2', 'day2', 'day2'],
        #         'Entries': [5.1, 6.2, 5.5, 6.0, 5.8, 6.1]}
        #     df = pd.DataFrame(exampledata)

    def models(self):
        self.models=[]
        self.results=[]
        for table in self.tables:
            model = mixedlm("Entries~Session*Group",table, groups=table["Subject"]) # Need to get rid of hard coding here.... 
            result = model.fit()

            # Append current model and results to attribute
            self.models.append(model)
            self.results.append(result)

            ipdb.set_trace()

if __name__=='__main__':
    # Parse command line inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_directory',type=str,required=True)
    parser.add_argument('--force', action='store_true', help="Delete previos objects and recalculate them")
    args=parser.parse_args()

    # Delete previously made objects
    if args.force:
        delete_saved_objects(root_dir=args.root_directory)

    # Set up main object 
    primaryobject=openfield_pipeline(root_dir=args.root_directory)

    # Run main object
    primaryobject()
 
    
#C:\Users\listo\Downloads\test_data_2P_head_fixed