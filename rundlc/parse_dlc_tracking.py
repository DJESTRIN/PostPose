#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steps in DLC pipeline run after analyzing the video.
"""
import argparse
import deeplabcut

def dlc_generate_detection_data(path_to_config, path_to_MP4):
    try:
        deeplabcut.convert_detections2tracklets(config=path_to_config,videos=[path_to_MP4])
        try:
            deeplabcut.stitch_tracklets(config_path=path_to_config,videos=[path_to_MP4])
            try:
                deeplabcut.create_labeled_video(config=path_to_config,videos=[path_to_MP4])
            except:
                print("Creating labeled video failed for video:")
                print(path_to_MP4)
        except:
            print("stitching tracklets failed for video:")
            print(path_to_MP4)
    except:
        print("Converting detections 2 tracklets failed for video:")
        print(path_to_MP4)

parser=argparse.ArgumentParser()
parser.add_argument('--config_file_dir',type=str,required=True) #input to config file
parser.add_argument('--MP4_file_dir',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    dlc_generate_detection_data(args.config_file_dir,args.MP4_file_dir)
