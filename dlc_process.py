#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process videos based on trained dlc model
"""
import argparse
import deeplabcut

def dlc_process(path_to_config, path_to_MP4):
    print("got here")
    deeplabcut.analyze_videos(path_to_config,path_to_MP4)
    try:
        deeplabcut.create_labeled_video(path_to_config,path_to_MP4)
    except:
        print("Unable to create video. Check if h5 file exists")

parser=argparse.ArgumentParser()
parser.add_argument('--config_file_dir',type=str,required=True) #input to config file
parser.add_argument('--MP4_file_dir',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    dlc_process(args.config_file_dir,args.MP4_file_dir)
