#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: tailsuspension.py
Description: Analyse DLC pose estimation outputs with respect to our task.
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""

"""
Important note to be deleted:
(1) focus on looking at velocity
(2) Using threshold for velocity, determine when mouse is moving, and calculate the percent/total time mouse moves 
    versus is not moving. 
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
import cv2
import ipdb

class tailsus_graphics(graphics):
    """ tailsus graphics ... uses graphics class to calculate the following details:
    """
    def __call__(self):
        super().__call__()
        self.basic_metrics()
        #self.percent_time_moving()

    def basic_metrics(self):
        # Calculate cumulative distance
        self.total_distance=np.sum(self.digested_obj.av_distance)*self.digested_obj.cms_per_pixel*0.1 # In pixels ==> NEED TO CONVERT TO CM

        # Calculate average speed 
        self.average_speed=np.mean(self.digested_obj.av_speed)*self.digested_obj.cms_per_pixel*0.1 

        # Calculate average acceleration magnitude in inner vs outer circle
        self.average_acc=np.mean(self.digested_obj.av_acc_mag)*self.digested_obj.cms_per_pixel*0.1 # In pixels ==> NEED TO CONVERT TO CM

    #def percent_time_moving(self):
     #   ipdb.set_trace()

class tailsus_pipeline(pipeline):
    """ Updated pipeline class to accomodate tailsus_graphics class """
    def __call__(self):
        """ Main set of steps for current analysis. """
        # Loop over csv files, making sure all csv files have been processed.
        self.digestion_objs = []
        self.arena_objs = []
        self.graphics_objs=[]

        for csvfile in self.csv_files:
            # Exclude non tailsus csv files
            if 'TST' not in csvfile.lower() and 'tst' not in csvfile.lower():
                continue
            
            outputfile,_=csvfile.split('.cs')
            outputfile+='digestion.pkl'

            # Determine if object was already created on a previous run.
            if os.path.isfile(outputfile):
                obj_oh = digestion.load(outputfile)
            else:
                obj_oh = digestion(csv_file = csvfile,framerate=60,cms_per_pixel=0.245,rolling_window=60)
                obj_oh()
                obj_oh.save(outputfile)
            
            # Keep all digestion objects inside of a single attribute
            self.digestion_objs.append(obj_oh)

            # Get corresponding video file
            video_file = self.match_csv_to_video(csvfile)

            # Pull video size
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise IOError("Cannot open video file")
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Re-write Arena shape
            self.shapes = [[['polygon']]]
            self.shape_positions = [[[(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]]]

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
                graph_obj = tailsus_graphics.load(graphics_file)
            else:
                graph_obj = tailsus_graphics(digested_obj=obj_oh,
                                    arena_obj=arena_objoh,
                                    behavior_name='TST',
                                    drop_directory=self.drop_directory)
                graph_obj()
                #graph_obj.save(field_file)
            
            # Keep all graphics objects inside a single attribute
            self.graphics_objs.append(graph_obj)

class tailsus_statistics(tailsus_pipeline):
    """ Generate statistics
    Description: This class is meant to pull all of the important data from each digestion object across groups 
        and capture statistics for each group. 
    """
    def build_tables(self,dependent_variables=["number_entries_innercircle"],independent_variables=["cage","mouse","day","group"],export_csv=True,normalize=True,behavior_name='none'):
        """ Build tables -- this method takes dependent variables and generates an aggregated table of the dependent variable
            given the list of independent variables.
        """
        self.tables=[] # a list of all pandas tables

        for depend in dependent_variables: # Loop over dependent variables
            for i,digobjoh in enumerate(self.graphics_objs): # Loop over digestion objects and pull data
                if hasattr(digobjoh,depend):
                    missing_attributes = [attr for attr in independent_variables if not hasattr(digobjoh.digested_obj, attr)]
                    
                    if missing_attributes:
                        print("missing attributes:", missing_attributes)
                        raise("Missing independent variables as attributes, must look at your spelling or if independent variables not generated")
                    else:
                        attribute_values = [getattr(digobjoh.digested_obj, attr, None) for attr in independent_variables]
                        depend_values = getattr(digobjoh,depend)

                        # Append data to dataframe 
                        if i==0:
                            data_oh = {'cage':attribute_values[0],
                                       'animal':attribute_values[1],
                                       'day':attribute_values[2],
                                       'group':attribute_values[3], 
                                       f'{depend}':depend_values}
                            table_f=pd.DataFrame([data_oh])
                        else:
                            data_oh = {'cage':attribute_values[0],
                                       'animal':attribute_values[1],
                                       'day':attribute_values[2],
                                       'group':attribute_values[3], 
                                       f'{depend}':depend_values}
                            table_oh=pd.DataFrame([data_oh])
                            table_f = pd.concat([table_f, table_oh], ignore_index=True)
                else:
                    raise("Dependent variable is not an attribute")
                  
            self.tables.append(table_f) # Put all tables into a list

        for table in self.tables:
            table["subject"]=table["cage"]+table["animal"]

        # If true, export all tables as a csv file into the results folder. 
        if export_csv:
            for table in self.tables:
                table_name = table.columns[-1]

                # Quick correction that is a bandaid
                if table_name == 'subject':
                    table_name = table.columns[-2]
                    
                output_csv_file = os.path.join(self.drop_directory,f"{table_name}_{behavior_name}.csv")
                table.to_csv(output_csv_file)

        if normalize:
            for table in self.tables:
                table_name = table.columns[-2]
                # Calculate percent change from day 0
                day0_values = table[table['day'] == '0'].set_index('subject')[table_name]
                table[table_name] = table[table_name] / table['subject'].map(day0_values)

    def table_plots(self,xaxis='day',group='group'):
        for table in self.tables:
            # Get table data
            table_name = table.columns[-2]
            table_av = table.groupby(["day","group"]).agg(Mean=(table_name, "mean"),
                                             StandardError=(table_name, lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))).reset_index()

            # Set offsets for each group
            group_offsets = {"CORT": -1, "CONTROL": 1}
            group_colors = {"CORT": 'red', "CONTROL": 'blue'}

            # Create figure
            plt.figure(figsize=(10, 10))

            # Plot bars with offsets according to group
            for group in table_av["group"].unique():
                group_data = table_av[table_av["group"] == group]
                offset = group_offsets[group]
                coloroh = group_colors[group]
                plt.bar(
                    group_data[xaxis].astype(float).values + offset, 
                    group_data["Mean"], 
                    yerr=group_data["StandardError"], 
                    capsize=5, 
                    color=coloroh, 
                    edgecolor='black',
                    width=1,          # Set bar width
                    alpha=0.6,         # Set bar transparency
                    error_kw={'elinewidth': 2},
                    label=f"Mean {group}"
                )

            # Plot individual subject data with corresponding offsets and colors
            for subject, subject_data in table.groupby("subject"):
                group = subject_data["group"].iloc[0]
                offset = group_offsets[group]
                color_oh = "red" if group == "CORT" else "blue"
                
                # Scatter plot points with the same offset as the bars
                plt.scatter(
                    subject_data[xaxis].astype(float).values + offset, 
                    subject_data[table_name], 
                    color=color_oh, 
                    label=group if subject == table["subject"].iloc[0] else ""  # Add legend once per group
                )

            # Add labels and formatting
            plt.xlabel(xaxis)
            plt.ylabel(f"Mean {table_name} ± Standard Error")
            plt.title(f"Mean {table_name} with Error Bars")
            plt.legend()
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            output=os.path.join(self.drop_directory,f'{table_name}_averages.jpg')
            plt.savefig(output)

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
    primaryobject=tailsus_statistics(root_dir=args.root_directory)

    # set shapes 
    shapesoh, shapestringsoh = 0,1 # place holder for now
    primaryobject.set_shapes(shape_positions=shapesoh,shapes=shapestringsoh)

    # Run main object
    primaryobject()

    # Build data tables
    primaryobject.build_tables(dependent_variables=["total_distance",
                                                    "average_speed",
                                                    "average_acc"],
                                                    normalize=False,
                                                    behavior_name='TST')

    # Generate plots for tables
    primaryobject.table_plots()
    print('Finished TST analysis')