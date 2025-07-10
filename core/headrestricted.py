#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: headrestricted.py
Description: For analysing head restricted tasks on the two-photon. Primarily will be used for analysing nose/ body movement during time-locked
    stimuli. 
Author: David Estrin
Version: 2.0
Date: 07-09-2025
"""
from gestion import digestion
from main import pipeline, delete_saved_objects
from graphics import graphics, experimental_field
import argparse
import numpy as np
import os, re, glob
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import pandas as pd
import ipdb

# Create headfixed object with general info
# Get timestamps from sync and sens files
# Update metrics using the timestamps. 
# Generate arena metrics
# Get arena information in class
# Update graphics with new timestamp metric data
    # Average velocity and distance after each trial type
    # raster-peth of velocity before and after each trial
# Run graphics using arena and headfixed object
# Run statistics to get tables for each behavior of interest


def grab_behavior_timestamps(path):
    """ Find corresponding sync and sens files for current csv file
        After grabbing the behavioral files, this function will parse out timestamps for 
        the various odors in task. Allowing analysis of video wrt to behavior. """
    
    ipdb.set_trace()

class headfixed_pipeline(pipeline):
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
                obj_oh = digestion(csv_file=csvfile,
                                   framerate=60,
                                   cms_per_pixel=0.245,
                                   rolling_window=60)
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
                graph_obj = graphics.load(graphics_file)
            else:
                graph_obj = graphics(digested_obj=obj_oh,
                                    arena_obj=arena_objoh,
                                    drop_directory=self.drop_directory)
                graph_obj()
                graph_obj.save(field_file)
            
            # Keep all graphics objects inside a single attribute
            self.graphics_objs.append(graph_obj)



class openfield_statistics(headfixed_pipeline):
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


if __name__=='__main__':

    all_csv_files = glob.glob(os.path.join(r'C:\Users\listo\tmt_experiment_2024_working_file\headfixed','*.csv'))
    digestion_objs = []
    for csv_file_oh in all_csv_files:
        obj_oh = digestion(csv_file=csv_file_oh)
        obj_oh()
        digestion_objs.append(obj_oh)




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
    primaryobject=openfield_statistics(root_dir=args.root_directory)

    # Run main object
    primaryobject()