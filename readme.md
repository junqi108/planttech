
<!-- ### *For a full documentation see:* [docs](https://qmxp55.github.io/planttech) -->
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

Remember to change the last line if needed with the correct path of your anaconda/miniconda distro:
```
prefix: /Users/tardis/opt/anaconda3/envs/plant-env
```

> If getting the issue `ResolvePackageNotFound` just move these packages under 'pip'. Install `laspy` with
> ```
> $ conda install -c conda-forge laspy
> ```

> If getting issue `No LazBackend selected, cannot decompress data` simply install:
> ```
> $ pip install laszip
> ```


## create kernel for env

Before login to our new conda env `plant-env`, we download the `ipykernel` repo,
```
pip install ipykernel
```

then, we create a kernel with based on the `plant-env` env,

```
python -m ipykernel install --user --name=plant-env 
```

Finally, we make sure it was created with:

```
jupyter kernelspec list 
```
