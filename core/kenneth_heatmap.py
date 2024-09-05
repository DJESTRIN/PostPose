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


    # Plot the image 
    plt.figure(figsize = (10,10))
    plt.imshow(image)
    plt.axis('off')  # Optional: Hide the axis

    # Plot the trajectory as line 
    plt.plot(trajectory[:,0],trajectory[:,1])


    plt.savefig(drop_file)

    #Generate a multiplot figure that has 2 colums and 2 tows and the first figure starts in the upper left. 
    fig, axs = plt.subplot(2, 2, 1)
    
    plt.subplot(2, 2, 1)
    axs[0,0].imshow(image)

    plt.subplot(2, 2, 2)
    axs[0,1].plot(trajectory)

    plt.subplot(2, 2, 3)
    #code for heatmap

    plt.subplot(2, 2, 4)
    # code combined 
    return  

if __name__=='__main__':
    trajectoryoh=np.random.randint(1,200,size=(100,2))
    image_dir=r'C:\Users\listo\PostPose\test_data\24-7-2_C4478776_M2.tif'
    imageoh=Image.open(image_dir)
    plot_heatmap(image=imageoh, trajectory=trajectoryoh, drop_file=r'C:\Users\listo\PostPose\test_data\example.tif')
        
