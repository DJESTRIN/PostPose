import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import ipdb


def plot_heatmap(image,trajectory,drop_file): #Image is numpy array, trajectory is a numpy array (x and y coordinates for mouse)
    # plot the image,
    # plot the trajectory as a line
    # plot heatmap of the trajectory as a heatmap. 
        # There is a function for this...
    # Make sure coloring is consistent for all mice ... 
    # Save this as an image in a output folder.
    
    # Plot the image 
    plt.figure(figsize = 10)
    plt.imshow(image)
    plt.axis('off')  # Optional: Hide the axis

    # Plot the trajectory as line 
    plt.plot(trajectory[:,0],trajectory[:,1])

    # Plot Heatmap
    plt.imshow(trajectory, cmap='hot', alpha = 0.5,  interpolation= 'nearest')

    plt.savefig(drop_file)
    return  
        
