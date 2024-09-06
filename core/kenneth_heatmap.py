import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import ipdb
import numpy as np
from PIL import Image
import matplotlib

def plot_heatmap(image,trajectory,drop_file, alpha = 2, beta = 0): 
    """ plot heatmap
    ## add description later
    """
    fig, axs = plt.subplots(2,2,figsize=(20,20),dpi=300)
    
    # Image of arena
    plt.subplot(2, 2, 1)
    image = np.round(np.array(image)*alpha+beta).astype(np.uint8)
    image = np.clip(image, 0, 255)
    axs[0,0] = plt.imshow(image)

    # Trajectory
    plt.subplot(2, 2, 2)
    axs[0,1] = plt.plot(trajectory[:,0],trajectory[:,1])
    # axs[0, 1].sharex(axs[0, 0])
    # axs[0,1]

    # Image of arena plus heatmap
    plt.subplot(2, 2, 3)
    axs[1,1] = plt.imshow(image)
    axs[1,1].hist2d(trajectory[:,0],trajectory[:,1], bins=100, norm=matplotlib.colors.LogNorm())

    # Image of everything combined
    plt.subplot(2, 2, 4)
    axs[0,1] = plt.plot(trajectory[:,0],trajectory[:,1])

    plt.savefig(drop_file)
    return  

if __name__=='__main__':
    trajectoryoh=np.random.randint(1,200,size=(100,2))
    image_dir=r'C:\Users\Kenneth Johnosn\PostPose\test_data\24-7-2_C4478776_M2.tif'
    imageoh=Image.open(image_dir)
    plot_heatmap(image=imageoh, trajectory=trajectoryoh, drop_file=r'C:\Users\Kenneth Johnosn\PostPose\test_data\example.tif')
       
