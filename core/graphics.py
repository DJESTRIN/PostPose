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
from skimage.draw import polygon
import matplotlib.pyplot as plt
import os
import pickle
import re
import ipdb

class experimental_field:
    def __init__(self,input_video,drop_directory,shape_positions=None,shapes=None):
        """ Experimental_field
        Inputs:
            shape_positions -- A list containing the positions for each individual shape. Shape positions are to be included using
                the following format.
                    Circle -- three values are expected per circle (X,Y,R). An X and Y midpoint coordinate and R radius.
                    polygon -- For all other shapes (triangle, rombus, octagon, etc) use the polygon option. For the polygon option,
                        include the X and Y coordinates for each corner of the shape (X1,Y1,X2,X2 ... (Xn,Yn))

            shapes -- A string or list of strings based on the following selection: 'circle','polygon'.
                For each shape, a certain number of vertices are expected inside of the shape positions array. 
        
        Intermediates:
            self.field_masks -- For every given shape, a mask is created for easy parsing to determine whether an animal's coordinates
                lay inside or outside the field of interest. Masks are arrange in the same order shapes are arranged (Ex. circle 1 corresponds
                with the first mask).

        Outputs: 
            field_image -- an example image of the field
        """

        # Set up class attributes
        self.input_video=input_video
        self.shapes=shapes
        self.shape_positions=shape_positions
        self.drop_directory=drop_directory

        # Set up the class
        self.arena_image=self.get_example_image(video_dir=input_video)
        self.objectnames(video_dir=input_video)
        self.__str__() # Get string name and print it
        self.build_experimental_field()

    def __str__(self):
        stringoh = f'CAGE{self.cage}_MOUSE{self.mouse}'
        self.string=stringoh
        return stringoh

    def objectnames(self,video_dir,pattern = r'day-(\d+)_C(\d+)_M(\d+)'):
        matches = re.search(pattern, video_dir)
        if matches:
            self.day = matches.group(1) # Extracts day 
            self.cage = matches.group(2)  # Extracts C4478776
            self.mouse = matches.group(3)  # Extracts M2
            if '_cort_' in video_dir:
                self.group='cort'
            else:
                self.group='control'
        else:
            print('problem')

    def get_example_image(self,video_dir=r'C:\Users\listo\PostPose\core\test_data\24-7-2_C4478776_M2.tif',frame_number=100):
        """ Given a video directory, open up an example image """
        # Determine if directory is actually an image or a video
        if video_dir.endswith(('.tiff','.tif','.jpg')):
            print('Building experimental field with image, not a video')
            example_image=Image.open(video_dir)
            example_image=np.array(example_image)

        # If directory is a video, use openCV
        else:
            try:
                cap = cv2.VideoCapture(video_dir)
                if not cap.isOpened():
                    raise("Error: Could not open video file")
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    _, example_image = cap.read()
                    cap.release()
            except:
                # Output None if video breaks
                example_image=None
        
        return example_image

    def build_experimental_field(self):
        """ Take shape data and convert it to an array of masks"""
        # Custom function for creating a mask given a shape
        def shape_to_mask(image,shape_coordinates,type):
            # Create an empty image
            height,width,depth=image.shape
            output_image=np.zeros((height,width),dtype=np.uint8)

            # Determine if the shape is a circle 
            if type[0]=='circle':
                (center_x, center_y, radius) = shape_coordinates 
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                output_image[mask] = 1 
            # Every other polygon
            elif type[0]=='polygon':
                # Parse vertices
                (xs,ys) = shape_coordinates
                allx, ally = polygon(xs, ys)
                output_image[allx, ally] = 1
            # Throw error
            else:
                raise('This shape is not included for our code or may be misspelled')
            
            return output_image

        def quick_plot_field(image, shape_masks, drop_file, alpha=0.4):
            for i,shape_mask in enumerate(shape_masks):
                # Convert mask to red
                (height, width) = shape_mask.shape
                red_image = np.zeros((height, width, 3), dtype=np.uint8)
                red_image[..., 0] = shape_mask * 255 * alpha

                if i==0:
                    fullmask=red_image
                else:
                    fullmask+=red_image

            # convert main image to color
            if len(image.shape) == 2:
                color_image = np.stack((image,)*3, axis=-1)
            else:
                color_image = image

            # Add main arena image and mask image together
            final_image = color_image + fullmask

            # Generate figure
            plt.figure(figsize=(10,10))
            plt.imshow(final_image)
            plt.axis('off')
            plt.savefig(drop_file)
            return 

        # Determine if arena image exists
        if self.arena_image is None:
            print("Cannot build arena without example image")

        else:
            # Determine if all necessary information for shapes was given
            if (self.shape_positions is not None) or (self.shapes is not None):
               # Gather mask images as attribute
               self.shape_masks=[]
               for (typeoh,shape_coordinatesoh) in zip(self.shapes,self.shape_positions):
                   self.shape_masks.append(shape_to_mask(self.arena_image,shape_coordinates=shape_coordinatesoh,type=typeoh))

               # Plot masks and arena image
               drop_fileoh = os.path.join(self.drop_directory,f'{self.string}_arena_mask_image.tif')
               quick_plot_field(self.arena_image,self.shape_masks,drop_file=drop_fileoh)

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
    def __init__(self,digested_obj,arena_obj,drop_directory=[]):
        # Set up attributes
        self.digested_obj=digested_obj #Get the digestion object
        self.arena_obj=arena_obj # Get the arena object created with experimental field class

        # Determine where figures will be dropped
        if drop_directory:
            self.drop_directory=drop_directory #Get the drop directory for figures
        else:
            self.drop_directory=self.digested_obj.drop_directory # Use the same drop directory inside of the digestion object

        # Determine if video_file was attached
        if hasattr(self.arena_obj,'arena_image'):
            self.attached_video=True
            self.arena_image=self.arena_obj.arena_image
        else:
            self.attached_video=False

    def __call__(self):
        self.correct_trajectories() # Correct trajectories
        self.plot_trajectory_and_heatmap() # Need to code these in later
        self.plot_metrics() # Plot common metrics

    def correct_trajectories(self):
        """ Correct trajectories - Using the arena image information, this method cuts off 
            data points outside of image range. """
        max_height,max_width,_ = self.arena_obj.arena_image.shape

        newxs=[]
        newys=[]
        for bpx,bpy in zip(self.digested_obj.x.T,self.digested_obj.y.T):
            bpx = np.where((bpx < 0) | (bpx > max_width), np.nan, bpx)
            bpy = np.where((bpy < 0) | (bpy > max_height), np.nan, bpy)
            newxs.append(bpx)
            newys.append(bpy)

        # Save over x and y digested attributes
        self.digested_obj.x=np.asarray(newxs).T
        self.digested_obj.y=np.asarray(newys).T

    def plot_trajectory_and_heatmap(self,alpha=2,beta=0):
        # If video is attached, pull and example image using random from the midle of the video
        # Plot the image, if no image, skip
        # Plot the x and y coordinates over the image

        # Generate figure
        fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10),dpi=200,constrained_layout=True)

        # Plot image of arena
        image = self.arena_obj.arena_image  # Get arena image
        image = np.round(np.array(image)*alpha+beta).astype(np.uint8) # change brightness/contrast
        image = np.clip(image, 0, 255) # change brightness
        max_x,max_y,_=image.shape
        axs[0,0].imshow(image)

        for spine in axs[0,0].spines.values():
            spine.set_visible(False)

        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,0].set_xticklabels([])
        axs[0,0].set_yticklabels([])

        # Plot trajectory
        axs[0,1].plot(self.digested_obj.x_av,self.digested_obj.y_av,color='black')
        axs[0,1].set_xlim(0,max_x)
        axs[0,1].set_xlim(0,max_y)

        for spine in axs[0,1].spines.values():
            spine.set_visible(False)
        
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,1].set_xticklabels([])
        axs[0,1].set_yticklabels([])

        # Plot heatmap
        axs[1,0].hist2d(self.digested_obj.x_av,self.digested_obj.y_av, bins=30,cmin=1, cmap='plasma')
        axs[1,0].set_xlim(0,max_x)
        axs[1,0].set_xlim(0,max_y)

        for spine in axs[1,0].spines.values():
            spine.set_visible(False)
        
        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])
        axs[1,0].set_xticklabels([])
        axs[1,0].set_yticklabels([])

        # Combine all plots
        # im=axs[1,1].imshow(image)
        hb=axs[1,1].hist2d(self.digested_obj.x_av,self.digested_obj.y_av, bins=30,cmin=1, cmap='plasma')
        axs[1,1].plot(self.digested_obj.x_av,self.digested_obj.y_av,alpha=0.5,color='black')
        im=axs[1,1].imshow(image)
        axs[0,1].set_xlim(0,max_x)
        axs[0,1].set_xlim(0,max_y)

        for spine in axs[1,1].spines.values():
            spine.set_visible(False)

        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        axs[1,1].set_xticklabels([])
        axs[1,1].set_yticklabels([])
    
        plt.tight_layout()
        print(self.digested_obj)
        output_path = os.path.join(self.drop_directory,f'{self.digested_obj.string}_heatmaps.jpg')
        plt.savefig(output_path)

    def plot_metrics(self,downsample=1):
        """ Generates a figure for the distance, speed and 
        acceleration magnitude for current gestation object """
        # Generate figure
        plt.figure(figsize=(10,10))
        
        # Plot distance
        plt.subplot(3,1,1)
        plt.plot(self.digested_obj.av_distance[::downsample])

        # Plot distance
        plt.subplot(3,1,2)
        plt.plot(self.digested_obj.av_speed[::downsample])

        # Plot acceleration magnitude
        plt.subplot(3,1,3)
        plt.plot(self.digested_obj.av_acc_mag[::downsample])

        # save file
        print(self.digested_obj)
        output_path = os.path.join(self.drop_directory,f'{self.digested_obj.string}_metricsgraph.jpg')
        plt.savefig(output_path)
    
    @classmethod
    def load(cls,filename):
        """Load an instance from a pickle file."""
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    def save(self,filename):
        """Save the instance to a file using pickle."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

