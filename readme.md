
# Outline

The following include the process to extract the `Leaaf Area Density` (LAD) from a mock point cloud generated in `Blensor` software. This process is splitted in 3 main cores:

- **LiDAR simulation on mockup trees with BLENSOR**
- **Data structure**
- **Leaf Inclination Angle (LIA) estimation**
- **Leaf Area Density (LAD) estimation**

# Preparation of data

In order to get the LIA and hence the LAD, we need to segmentated the trees and the leaves.

First, we define the name of the directory where the Blensor output data is, in this particular case we will look for directory `test`. Pipeline will look for this directory inside the `data` directory.