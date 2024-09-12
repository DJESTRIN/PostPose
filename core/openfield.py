#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: graphics.py
Description: 
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
import ipdb

class openfield_graphics(graphics):
    """ Openfield graphics ... uses graphics class to calculate the following details:
        (1) Count number of times body (part) passes from inner to outer circle (vice versa). 
            Default is average of all body parts.
        (2) Percent of time body (part) spends inside inner circle versus outer circle. 
            Default is average of all body parts.
        (3) Average distance traveled, speed and acceleration magnitude of body (part) inside inner circle
            versus outer circle. Default is average of all body parts.
    """
    def __call__(self):
        super().__call__()
        self.calculate_percent_time()
        self.number_entries_innercircle = self.calculate_transitions_innercircle()
        self.circle_metrics()
        ipdb.set_trace()

    def is_inside_circle(self,x_trajectory,y_trajectory, x_center, y_center, radius):
        distances_squared = (x_trajectory - x_center) ** 2 + (y_trajectory - y_center) ** 2
        return distances_squared <= radius ** 2

    def calculate_percent_time(self,x=None,y=None):
        if (x is None) or (y is None): 
            x=self.digested_obj.x[:,1] ### NEED TO FIX THIS...
            y=self.digested_obj.y[:,1]
        
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

class openfield_pipeline(pipeline):
    """ Updated pipeline class to accomodate openfield_graphics class """
    def __call__(self):
        """ Main set of steps for current analysis. """
        # Loop over csv files, making sure all csv files have been processed.
        self.digestion_objs = []
        self.arena_objs = []

        for csvfile in self.csv_files:
            outputfile,_=csvfile.split('.cs')
            outputfile+='.pkl'

            # Determine if object was already created on a previous run.
            if os.path.isfile(outputfile):
                obj_oh = digestion.load(outputfile)
            else:
                obj_oh = digestion(csv_file=csvfile)
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

def generate_openfield_shapes(input_circle_shape,input_shape_string,percent=0.75):
    """ Generate openfield shapes """
    # Break down components of input shapes
    X,Y,R = input_circle_shape[0][0]
    stringoh = input_shape_string[0][0]

    # Get output shape components for smaller inner circle
    Rnew=np.round(R*percent).astype(np.uint8)

    # Re build output lists
    output_circle_shape = [X,Y,Rnew]
    input_circle_shape = [X,Y,R]
    input_shape_string = stringoh
    output_shape_string = stringoh
    return [[input_circle_shape,output_circle_shape]],[[input_shape_string,output_shape_string]]

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

    # set shapes
    shapesoh,shapestringsoh = generate_openfield_shapes(input_circle_shape=[[[360,260,200]]],input_shape_string=[[['circle']]])
    primaryobject.set_shapes(shape_positions=shapesoh,shapes=shapestringsoh)

    # Run main object
    primaryobject()