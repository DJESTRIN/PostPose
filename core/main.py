#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: main.py
Description: Contains the primary protocol for running the postpose core code. Searches for files and then runs them through appropriate set
    of steps. 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""

# Import dependencies and libraries
import argparse
import os,glob
from gestion import digestion
from graphics import experimental_field, graphics
import re
import difflib
import ipdb

class pipeline:
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.drop_directory=self.get_dropdirectory(root_dir=self.root_dir)

        # Get all files of interest
        self.video_files=self.find_files(extension='.mp4')
        self.csv_files=self.find_files(extension='.csv')
        self.custom_objects=self.find_files(extension='.pkl')
 
    def get_dropdirectory(self,root_dir):
        subdirectory_path = os.path.join(root_dir, 'results')
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
        return subdirectory_path

    def find_files(self,extension=".csv"):
        """ find all files of interest (csv, video, etc) in 
        current directory and put them in organized list """
        return glob.glob(f"{self.root_dir}/**/*{extension}", recursive=True)

    def match_csv_to_video(self,csv_file):
        original_videos = [s for s in self.video_files if 'resnet' not in s.lower()] # removes any of the labeled videos from list
        matches = difflib.get_close_matches(csv_file, original_videos, n=1)
        video_file=matches[0]
        return video_file
    
    def set_shapes(self,shape_positions,shapes):
        self.shape_positions=shape_positions
        self.shapes=shapes

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

class generate_statistics(pipeline):
    """ Generate statistics
    Description: This class is meant to pull all of the important data from each digestion object across groups 
        and capture statistics for each group. 
    """
    def __call__(self):
        # Inherit previous call method from pipeline
        super().__call__()

        # Loop over digestion objects and pull data
        for digobjoh in self.digestion_objs:
            a=1

def delete_saved_objects(root_dir):
    """ Delete save objects
    Find objects in root directory and then delete the pkl files. 
    """
    objsoh = glob.glob(os.path.join(root_dir,'*.pkl'))
    for objoh in objsoh:
        try:
            os.remove(objoh)
        except:
            print(f'The following object was not found: {objoh}')

if __name__=='__main__':
    # Parse command line inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_directory',type=str,required=True)
    args=parser.parse_args()

    # Set up main object 
    primaryobject=pipeline(root_dir=args.root_directory)

    # set shapes
    primaryobject.set_shapes(shape_positions=[[[360,260,200]]],shapes=[[['circle']]])

    # Run main object
    primaryobject()

    #C:\Users\listo\Downloads\test_data_for_videos