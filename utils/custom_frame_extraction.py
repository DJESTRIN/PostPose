#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:51:15 2023

@author: dje4001
"""
import cv2
import argparse

def extractframes(input_video_dir,output_image_dir):
    count=0
    vidcap=cv2.VideoCapture(input_video_dir)
    length=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    denomiator=round(length/20) # Get 20 frames per video
    success,image=vidcap.read()
    success = True
    while success:
        success,image=vidcap.read()
        if (count%denomiator)==0:
            cv2.imwrite(output_image_dir+"/img%d.png" % count,image)
        count+=1

if __name__=='__main__':
   a=argparse.ArgumentParser()
   a.add_argument("--input_video_dir", help="path to video")
   a.add_argument("--output_image_dir", help="path to images")
   args = a.parse_args()
   extractframes(args.input_video_dir,args.output_image_dir)


