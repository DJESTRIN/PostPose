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
import os
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
                graph_obj = graphics.load(graphics_file)
            else:
                graph_obj = graphics(digested_obj=obj_oh,
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
    args=parser.parse_args()

    # Delete previously made objects
    delete_saved_objects(root_dir=args.root_directory)

    # Set up main object 
    primaryobject=pipeline(root_dir=args.root_directory)

    # set shapes
    shapesoh,shapestringsoh = generate_openfield_shapes(input_circle_shape=[[[360,260,200]]],input_shape_string=[[['circle']]])
    primaryobject.set_shapes(shape_positions=shapesoh,shapes=shapestringsoh)

    # Run main object
    primaryobject()