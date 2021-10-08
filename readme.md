
### *For a full documentation see:* [docs](https://qmxp55.github.io/planttech)
# Outline

The following include the process to extract the `Leaf Inclination Angle` (LIA) with respect to zenith, the `Leaf Area Index` (LAI), and the `Leaf Area Density` (LAD) of a point cloud (PC). We test our pipeline using a mockup point cloud generated in `Blensor` software. This process is outlined below:

- **LiDAR simulation on mockup trees with BLENSOR**
    - Toy tree and sensor specifications
    - Data structure
    - Leaf Inclination Angle (LIA) estimation
    - Leaf Area Density (LAD) estimation
  
- **Aplication on Kiwkifruit LiDAR dataset**



# Instalation

When creating the list of requirements use:

```
conda env export -n <env-name> --no-builds> requirements.yml
```

and to create the environment use:

```
conda env create -f path/to/environment.yml
```

If getting the issue `ResolvePackageNotFound` just move these packages under 'pip'.
