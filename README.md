<h1> <b> 🐭 PostPose 🐭 </b> </h1> 
A repository for analyzing outputs of DeepLabCut, a pose-estimation software. This repository utalizes DLC's API for pose-estimation, and then anlayzes these outputs based on common experiments utilized in neuroscience. Originally, this pipeline was developed for SLURM-based clusters. However, the core code and scripts are meant to be run locally. 

<h2> <b> ⚠️ Warning: This code is still under development. ⚠️ </b> </h2>
Please kindly ignore any issues with code as well as any missing citations to others code. 

<h2> <b> Examples </b> </h2>
One can build an experimental arena in order to track specific regions of interest.
<p float="left">
  <img src="https://github.com/DJESTRIN/PostPose/blob/main/examples/example_arena.png" width="500" />
</p>

DLC-derived tracking data is then parsed in order to analyze various metrics in regions of interest.
<p float="left">
  <img src="https://github.com/DJESTRIN/PostPose/blob/main/examples/example_result.png" width="500" />
</p>

<h2> <b> References </b></h2>
Portions of this library utalize code from (or are inspired by) the following references:

- <b> Pose-Estimation: </b> Mathis, A., Mamidanna, P., Cury, K. M., Abe, T., Murthy, V. N., Mathis, M. W., & Bethge, M. (2018). DeepLabCut: Markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience, 21(9), 1281-1289. https://doi.org/10.1038/s41593-018-0209-y

<h2> <b> Contributions and citation </b> </h2>
David James Estrin & Kenneth Wayne Johnson contributed equally on this project.

- Code: David James Estrin, Kenneth Wayne Johnson
  
- Data: Kenneth Wayne Johnson, David James Estrin
