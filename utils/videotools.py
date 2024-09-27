#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: videotools.py
Description: Includes functions for cropping videos
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob,os
import tqdm
import ipdb

def deviation_image(video_file,skip=1):
    cap = cv2.VideoCapture(video_file)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flag=0
    images=[]
    for framen in range(frame_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, framen)
        _, image_oh = cap.read()
        image_oh = cv2.cvtColor(image_oh, cv2.COLOR_BGR2GRAY)
        image_oh=np.asarray(image_oh)
        images.append(image_oh)

    images = np.asarray(images)
    deviation_image = np.std(images[:10000,:,:],axis=0)
    return images, deviation_image.astype(np.uint8)

def detect_blobs(image,threshold=10,stop=5):
    # normalize image
    image_norm = (image-image.min())/(image.max()-image.min())

    # Get max values
    xhists=np.max(image_norm,axis=0)
    yhists=np.max(image_norm,axis=1)

    # find peaks of max values for each dimension
    # x dimension
    peaks, _ = find_peaks(xhists)
    xyvals=[]
    for peak in peaks:
        xyvals.append([xhists[peak],peak])

    # y dimension
    peaksy, _ = find_peaks(yhists)
    yyvals=[]
    for peak in peaksy:
        yyvals.append([yhists[peak],peak])
    
    yyvals.sort(reverse=True)
    xyvals.sort(reverse=True)
    xyvals=np.asarray(xyvals)
    yyvals=np.asarray(yyvals)

    current_list=[xyvals[0,1]]
    for y,x in xyvals[1:,:]:
        current_list=np.asarray(current_list)
        flag=False
        for xf in current_list:
            if abs(xf-x)<75:
                flag=True
        
        if not flag:
            current_list = [x for x in current_list]
            current_list.append(x)

        if len(current_list)>=stop:
            break
    current_list.sort()

    return current_list, yyvals[0][1]

def crop_around_coordinate(video_name,video,x,y,paddingx=50,paddingy=90,fpsoh=60, display=False):
    cropped_video = video[:,int(y-paddingy):int(y+paddingy),int(x-paddingx):int(x+paddingx)]
    # write cropped numpy array to video
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'),fpsoh,(cropped_video.shape[2],cropped_video.shape[1]))
    for frame in cropped_video:
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        if display:
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
    out.release()
    if display:
        cv2.destroyAllWindows()

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def generate_filenames(video_file,output_dir):
    # Get root info for all mice
    names=video_file.split('_M')
    root_info=names[0].split('\\')
    root_info=os.path.join(output_dir,root_info[-1])

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Build file names for each mouse into list
    midnames=names[1:-1]
    finname,_=names[-1].split('.m')
    running_list=[f'{root_info}_M{name}.mp4' for name in midnames]
    running_list.append(f'{root_info}_M{finname}.mp4') 
    ipdb.set_trace()
    return running_list 

def main(search_path=r'C:\Users\listo\Downloads\OneDrive_2_9-19-2024\redo',output_path=r'C:\Users\listo\Downloads\TST_02_09-19-2024_cropped'):
    search_string=os.path.join(search_path,'*.mp4')
    videos=glob.glob(search_string)

    for video_fileoh in tqdm.tqdm(videos):
        video_array, imageoh = deviation_image(video_file=video_fileoh)
        filenamelist=generate_filenames(video_file=video_fileoh,output_dir=output_path)
        xvals,yval = detect_blobs(imageoh,stop=len(filenamelist))

        for filename,x in zip(filenamelist,xvals):
            crop_around_coordinate(video_name=filename,video=video_array,x=x,y=yval,display=False)
    
    return 

if __name__=='__main__':
   main()