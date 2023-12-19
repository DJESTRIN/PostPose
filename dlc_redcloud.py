#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut Red Cloud Script
"""
import argparse
#import deeplabcut
import os,glob,sys
import pandas as pd
sys.path.append("/home/dje4001/post_dlc/") #folder to custom dlc code
from parse import hdf_to_tall
import concurrent.futures


# def train_dlc(path_to_config):
#     path_to_config=str(path_to_config)
#     deeplabcut.load_demo_data(path_to_config)
#     deeplabcut.train_network(path_to_config, shuffle=1,displayiters=5,saveiters=100)
#     deeplabcut.evaluate_network(path_to_config)
#     return

# def dlc_process(path_to_config, path_to_MP4):
#     try:
#         deeplabcut.analyze_videos(path_to_config,[path_to_MP4], auto_track=True,videotype='mp4')
#         deeplabcut.create_video_with_all_detections(path_to_config, [path_to_MP4], videotype='mp4')
#         deeplabcut.create_labeled_video(path_to_config,[path_to_MP4],color_by="individual",keypoints_only=False,
#                                         trailpoints=10,draw_skeleton=True,track_method="ellipse")
#     except:
#         print("Post-process analysis failed")     

# def dlc_generate_detection_data(path_to_config, path_to_MP4):
#     try:
#         deeplabcut.convert_detections2tracklets(config=path_to_config,videos=[path_to_MP4])
#         try:
#             deeplabcut.stitch_tracklets(config_path=path_to_config,videos=[path_to_MP4])
#             try:
#                 deeplabcut.create_labeled_video(config=path_to_config,videos=[path_to_MP4])
#             except:
#                 print("Creating labeled video failed for video:")
#                 print(path_to_MP4)
#         except:
#             print("stitching tracklets failed for video:")
#             print(path_to_MP4)
#     except:
#         print("Converting detections 2 tracklets failed for video:")
#         print(path_to_MP4)    
 
def num_sort(filename):
    # A function that sorts names correctly 
    broken_name=filename.split('_')
    print(broken_name)
    try:
        index=broken_name[19]
        index,_=index.split('.')
    except:
        index=broken_name[20]
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
        df=pd.read_hdf(file)
        final_data_frame.append(df)
    
    final_filename = get_filename(sorted_file_list)
    final_data_frame.to_csv(final_filename)

def group_videos(video_dir):
    #videos were originally split, 
    #this function takes the output h5 data 
    #and concatenates. 
    os.chdir(video_dir)
    starting_files=glob.glob('*1.*h5')
    
    for file in starting_files:
        location=file.find('1.')
        query_file=file.replace(file[(location-2):(location+2)],'*')
        print(query_file)
        related_files=set(glob.glob(query_file))
        ignore_files=set(glob.glob("*DLC*"))
        related_files=list(related_files-ignore_files)
        related_files.sort(key=num_sort)
        print(related_files)
        concatenate_dataframes(related_files)
    return

group_videos("/athena/listonlab/store/dje4001/deeplabcut/processed_video_drop/")


parser=argparse.ArgumentParser()
parser.add_argument('--config_file_dir',type=str,required=True) #input to config file
parser.add_argument('--parent_path_videos',type=str,required=True) #input to config file
parser.add_argument('--training',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    
    """ (1) Train if necessary """
    if args.training=="yes":
        print("Starting training")
        train_dlc(args.config_file_dir)
    else:
        print("No Training. Going right to analysis")

    """ (2) Loop over videos, pull them through DLC for labeling """
    #Get videos
    os.chdir(args.parent_path_videos)
    videos=glob.glob("*.mp4")
    
    for video in videos:
        dlc_process(args.config_file_dir,video)
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future_to_dlcprocess= {executor.submit(dlc_process,args.config_file_dir,video): video for video in videos}

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future_to_dlcprocess= {executor.submit(dlc_generate_detection_data,args.config_file_dir,video): video for video in videos}

    # """ (3) Takes output of dlc_process and concatenates videos together into larger h5 files """
    # group_videos(args.parent_path_videos)

    # """ (4) Parse h5 files and output as csv files """
    # h5_files=glob.glob("*.h5")
    # for h5oh in h5_files:
    #     hdf_to_tall(h5oh)
        
    # """ (5) Concatonate all csv files into a final tall format """
    # all_files=[i for i in glob.glob('*.csv')]
    # combined_csv=pd.concat([pd.read_csv(f) for f in all_files])
    # combined_csv.to_csv("",index=False)