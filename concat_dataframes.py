#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concat output of dlc analyze into bigger h5 files. 1 video per animal per day
"""
import os,glob
import pandas as pd
import argparse
 
def num_sort(filename):
    # A function that sorts names correctly 
    broken_name=filename.split('_')
    index=broken_name[19]
    index,_=index.split('.')
    number=int(''.join(c for c in index if c.isdigit()))
    return number

def get_filename(file_list):
    # Generate a final file name for the concatenated data
    file=file_list[0]
    file,extension=file.split('.')
    file=file+"CONCATENATED"+extension
    return file
    
def concatenate_dataframes(sorted_file_list):
    # concatenate data frames from all of the sorted files
    final_data_frame=[]
    for file in sorted_file_list:
        df=pd.read_h5(file)
        final_data_frame.append(df)
    
    final_filename = get_filename(sorted_file_list)
    final_data_frame.to_csv(final_filename)

def group_videos(video_dir):
    #videos were originally split, 
    #this function takes the output h5 data 
    #and concatenates. 
    os.chdir(video_dir)
    starting_files=glob.glob('*1.*mp4')
    
    for file in starting_files:
        location=file.find('1.')
        query_file=file.replace(file[(location-2):(location+2)],'*')
        related_files=glob.glob(query_file)
        related_files.sort(key=num_sort)
        concatenate_dataframes(related_files)
    return

""" video directory to start concating data """
parser=argparse.ArgumentParser()
parser.add_argument('--videos_dir',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    os.chdir(args.videos_dir)
    group_videos(args.videos_dir)
