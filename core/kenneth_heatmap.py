import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import ipdb
import numpy as np
from PIL import Image

def plot_heatmap(image,trajectory,drop_file): #Image is numpy array, trajectory is a numpy array (x and y coordinates for mouse)
    # plot the image,
    # plot the trajectory as a line
    # plot heatmap of the trajectory as a heatmap. 
        # There is a function for this...
    # plot shape if there is a shape...
    # Make sure coloring is consistent for all mice ... 
    # Save this as an image in a output folder.
    # figure out how to keep x and y range using same cordinate system as image. 

    

    #Generate a multiplot figure that has 2 colums and 2 tows and the first figure starts in the upper left. 
    plt.figure()
    fig, axs = plt.subplots(2,2)
    fig, axs = plt.subplots(2,2,figsize=(20,20),dpi=300)
   # plt.subplots(figsize=(20,20),dpi=300)
    plt.subplot(2, 2, 1)
    image = np.round(np.array(image)*alpha+beta).astype(np.uint8)
    image = np.clip(image, 0, 255)
    axs[0,0] = plt.imshow(image)

    plt.subplot(2, 2, 2)
    axs[0,1] = plt.plot(trajectory[:,0],trajectory[:,1])
    axs[0, 1].sharex(axs[0, 0])
    axs[0,1]
    plt.subplot(2, 2, 1)
    axs[0,0] = plt.imshow(image)

    plt.subplot(2, 2, 2)
    axs[0,1] = plt.plot(trajectory[:,0],trajectory[:,1])

    #dave
    #plt.subplot(2, 2, 3)
    #code for heatmap

    #plt.subplot(2, 2, 4)
    # code combined 
    plt.savefig(drop_file)
    return  

if __name__=='__main__':
    trajectoryoh=np.random.randint(1,200,size=(100,2))
    image_dir=r'C:\Users\Kenneth Johnosn\PostPose\test_data\24-7-2_C4478776_M2.tif'
    imageoh=Image.open(image_dir)
    plot_heatmap(image=imageoh, trajectory=trajectoryoh, drop_file=r'C:\Users\Kenneth Johnosn\PostPose\test_data\example.tif')
       
