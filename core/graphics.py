#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: graphics.py
Description: 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""
from PIL import Image
import cv2 
import numpy as np

class experimental_field:
    def __init__(self,shape_positions=None,shapes=None):
        """ Experimental_field
        Inputs:
            shape_positions -- A list containing the positions for each individual shape. Shape positions are to be included using
                the following format.
                    Circle -- three values are expected per circle (X,Y,R). An X and Y midpoint coordinate and R radius.
                    Rectangle -- four values are expected per rectangle (X1,Y1,X2,Y2). 
                    polygon -- For all other shapes (triangle, rombus, octagon, etc) use the polygon option. For the polygon option,
                        include the X and Y coordinates for each corner of the shape (X1,Y1,X2,X2 ... (Xn,Yn))

            shapes -- A string or list of strings based on the following selection: 'circle','rectangle','polygon'.
                For each shape, a certain number of vertices are expected inside of the shape positions array. 
        
        Intermediates:
            self.field_masks -- For every given shape, a mask is created for easy parsing to determine whether an animal's coordinates
                lay inside or outside the field of interest. Masks are arrange in the same order shapes are arranged (Ex. circle 1 corresponds
                with the first mask).

        Outputs: 
            field_image -- an example image of the field
        """

        # Set up class attributes
        self.shapes=shapes
        self.shape_positions=shape_positions

        # Set up the class
        self.arena_image=self.get_example_image()
        self.build_experimental_field()
    
    def get_example_image(self,video_dir=r'C:\Users\listo\PostPose\core\test_data\24-7-2_C4478776_M2.tif',frame_number=100):
        """ Given a video directory, open up an example image """
        # Determine if directory is actually an image or a video
        if video_dir.endswith('.tiff','.tif','.jpg'):
            print('Building experimental field with image, not a video')
            example_image=Image.open(video_dir)
            example_image=np.array(example_image)

        # If directory is a video, use openCV
        else:
            cap = cv2.VideoCapture(video_dir)
            if not cap.isOpened():
                raise("Error: Could not open video file")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, example_image = cap.read()
                cap.release()

        return example_image

    def build_experimental_field(self):
        """ Take shape data and convert it to an array of masks"""
        # Is there an example image for experiment
        # Are there any shapes?
        # Make sure shape sizes match ... each circle has a midpoint...
    
        def shape_to_mask(image,shapeoh):
            # Convert example image to a mask for current shape.
            # 
        
        #  

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
        #
    
