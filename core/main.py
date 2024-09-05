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

class main:
    def __init__(self,root_dir):
        self.root_dir=root_dir

        # Get all files of interest
        self.video_files=self.find_files(extension='.mp4')
        self.csv_files=self.find_files(extension='.csv')
        self.custom_objects=self.find_files(extension='.pkl')

    def find_files(self,extension=".csv"):
        """ find all files of interest (csv, video, etc) in 
        current directory and put them in organized list """
        return glob.glob(f"{self.root_dir}/**/*{extension}", recursive=True)

    def __call__(self):
        """ Main set of steps for current analysis. """
        # Run digestion, or tst, or openfeild
        # Run graphics
        # Collect info?

        # Determine if csv file has been analyzed before, load in data if true
        if os.path.isfile(outputfile):
            obj_oh = digestion.load(outputfile)
        else:
            obj_oh = digestion(inputfile)
            obj_oh()
            obj_oh.save(outputfile)

        # Set up experimental arena
        arena_obj = experimental_field()
        
        # Generate graphics for current obj ... add in a loading feature later. 
        graph_obj = graphics(digestiondata=obj_oh)


if __name__=='__main__':
    # Parse command line inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_directory',type=str,required=True)
    args=parser.parse_args()

    # Set up main object 
    parimaryobject=main(root_dir=args.root_directory)

    # Run main object
    parimaryobject()