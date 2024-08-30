#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: graphics.py
Description: 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""

class build_experimental_field:
    # Does the experiment include regions of interest (circles or squares)?
        # Where are they and label them
    # Grab an image of the experiment 

class graphics():
    def __init__(self,digested_obj,drop_directory=[],video_file=[]):
        self.objoh=digested_obj #Get the digestion object
        # Determine where figures will be dropped
        if drop_directory:
            self.drop_directory=drop_directory #Get the drop directory for figures
        else:
            self.drop_directory=self.objoh.drop_directory # Use the same drop directory inside of the digestion object

        # Determine if video_file was attached
        if video_file:
            self.video_file=video_file
            self.attached_video=True
        else:
            self.attached_video=False

    def graphics_pipeline(self):
        # Need to code these in later
        self.plot_trajectory()
        self.plot_distance()
        self.plot_speed()
        self.plot_acceleration()

    def plot_trajectory(self):
        # If video is attached, pull and example image using random from the midle of the video
        # Plot the image, if no image, skip
        # Plot the x and y coordinates over the image
        a=1

    def plot_heatmap(self):
        # Take method from plot_trajectory,
        # bin the pixels for the heatmap
        # plot heatmap instead of the line to show where the mouse generally was located on screen
    
