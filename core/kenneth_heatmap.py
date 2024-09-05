import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import ipdb


def plot_heatmap(image,trajectory,drop_file): #Image is numpy array, trajectory is a numpy array (x and y coordinates for mouse)
    # plot the image,
    # plot the trajectory as a line
    # plot heatmap of the trajectory as a heatmap. 
        # There is a function for this...
    # plot shape if there is a shape...
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

    #Generate a multiplot figure that has 2 colums and 2 tows and the first figure starts in the upper left. 
    fig, axs = plt.subplot(2, 2, 1)
    axs[0,0].plot(image)
    axs[0,1].plot(trajectory)
    axs[1,0].plot() #heatmap
    axs[1,1].plot() #heatmap overlay


    return  
        
