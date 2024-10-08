from videotools import deviation_image, detect_blobs, crop_around_coordinate
import multiprocessing as mp
import os,glob
import ipdb
import matplotlib.pyplot as plt
import numpy as np

def crop(file):
    _,devimage=deviation_image(file,stop=1000) # Get deviation image

    # Get blob coordinates
    xval,yval = detect_blobs(devimage,stop=3)
    xval=np.asarray(xval)
    xval=xval.mean()

    # Set up output file
    output_filename, _ = file.split('.mp')
    output_filename += '_cropped.mp4'

    # Do the cropping
    crop_around_coordinate(video_name=output_filename,video=file,x=xval,y=yval,display=False,paddingx=250,paddingy=250)

def main(root_dir):
    # Search for videos
    all_videos = [f for search_pattern in [os.path.join(root_dir,'**/*.mp4'), os.path.join(root_dir,'**/*.avi')] for f in glob.glob(search_pattern,recursive=True)]
    head_fixed_videos = [file for file in all_videos if (('_side' in file) or ('_front' in file)) and ('_cropped' not in file)]

    # Run crop function with threading
    with mp.Pool(processes=mp.cpu_count()) as p:
        p.map(crop, head_fixed_videos)

if __name__=='__main__':
    main(r'C:\Users\listo\Downloads\head_fixed_videos')