# `Leaf Inclination Angle` (LIA) estimation

Most functions used in this chapter are in library:

``` Python
import lia
```

Outputs will be placed inside directory `lia`,

```
root
└───data
    └───test
        │   s0100000.numpy
        │   s0200000.numpy
        │   ...
        │   mesh.ply
        │   scanner_pos.txt
        └───lia
            │   angles_<treename>.npy
            │   weights_<treename>.npy
            |   leaf_angle_dist_<treename>.png
            |   leaf_angle_dist_height_<treename>.png
            |   bestfits_pars_treename>.png

```

## The method (`lia.leaf_angle()`)

The main function that computes the LIA is `lia.leaf_angle()` which uses a KDtree approximation. The steps are as follow: 

1. `Compute normals`: This method fits a plane based on the nearesth neighbors for each point and gets the normal of this plane.
    
2. `Compute zenith angles`: Then, using the dot product with get the angle with respect to the zenith (i.e. agains vector (0, 0, 1)) 
    
3. `Range correction`: The results angles run from $0 < \theta < 180$, however we require these to be in the range $0 < \theta < 90$ therefore we transfom those angles $> 90$ with relation:

$$\theta_{L} = 180 - \theta$$

4. `Weights correction`: The resulting LIA is biased to PC density and completeness. In order to reduce this biases, we compute weights via voxelization,

$$\eta_{i} = n_{i}/L^{3} \\
  \bar{\eta} = \frac{1}{N}\sum_{i=0}^{N} \eta{i}$$

where $n_{i}$ is the number of points within voxel $i$, $L$ is the voxel size, and $N$ is the total number of voxels, then $\eta_{i}$ is the volume density of voxel $i$, and $\bar{\eta}$ is the mean volume density.

The function `lia.leaf_angle()` has to be ran per tree and requieres 6 input parameters:

- `points`: $x$, $y$ and $z$ coordinates of the leaf point cloud (LPC).
- `mockname`: name of directory where the data is.
- `treename`: name/index of tree.
- `voxel_size_w`: voxel size for `weights correction` i.e. $L$.
- `kd3_sr`: KDtree searching radius for the nearest neighboors serch.
- `max_nn`: Maximum number of nearest neightbors to be considered.

This function returns a set of files inside directory `lia`:

- `angles_<treename>.npy`: LIA for the LPC. One file per tree.
- `weights_<treename>.npy`: LIA weights for the LPC. One file per tree.
- `leaf_angle_dist_<treename>.png`: Figure of LIA ($\theta_{L}$) distribution with `weights correction`. If `Truth` LIA available, this will be shown alongside. One figure per tree.
- `leaf_angle_dist_height_<treename>.png`: Top - Figure of LPC distribution accross different heights in terms of voxels $k$. Bottom - If `Truth` LIA available, $\theta_{L}^{truth} - \theta_{L}$. The different curves show this for different heights ($k$). One figure per tree.

## Look for best-fit `voxel_size_w`, `kd3_sr` and `max_nn` with `lia.bestfit_pars_la()`

If truth LIA available i.e. there's a mesh file `mesh.ply` in the `test` directory, then we will be able to run function `lia.bestfit_pars_la()` which essentialy runs `lia.leaf_angle()` for a range of values in `voxel_size_w`, `kd3_sr` and `max_nn` and find the best-fit for these three based on the minimal $\chi^{2}$ between the estimated LIA and the truth LIA.

`lia.bestfit_pars_la()` is as well ran per tree and requires only `points`, `mockname` and `treename`. It returns `bestfit_<treename>.npy` file that contains the `voxel_size_w`, `kd3_sr` and `max_nn` best-fit values per tree. it also returns a dictionary with the $\chi^{2}$ for each of these runs.

Using output dictionary from `bestfit_pars_la` we can run `bestfit_pars_la` to create figure `bestfits_pars_treename>.png` that shows the $\chi^{2}$ for all the ranges used in `voxel_size_w`, `kd3_sr` and `max_nn`.

```{admonition} To-Do
:class: important
Current LIA implementation works without `Truth` LIA, however, we need it to estimate the best-fits `voxel_size_w`, `kd3_sr` and `max_nn` parameters. We need to find the relation between these three and LPC that could rely on the LPC density, leaf size, leaf area, etc.
```

The piece of code bellow runs `lia.bestfit_pars_la()` and `lia.best_fit_pars_plot()` for each tree.

```Python
for key, val in trees.items():

    keep = (val) & (leaves) # take the LPC per tree
    df_ = df[['x', 'y', 'z']][keep]
    points = loads.DF2array(df_)
    res = lia.bestfit_pars_la(points, mockname, treename=key)
    lia.best_fit_pars_plot(res, key, mockname)
```

once we find the best-fit parameters we get figure `bestfits_pars_treename>.png` that is shown in Fig. {numref}`bestfits_pars`. we use these best fits to run `lia.leaf_angle()` and get the LIA and corresponding weigths per tree. The code that does that is shown below and in Fig. {numref}`lia_dist` we show `leaf_angle_dist_<treename>.png` and in Fig. {numref}`lia_dist_h` `leaf_angle_dist_height_<treename>.png`.


```{figure} ../gifs/bestfits_pars_tree_0.png
---
width: 30em
name: bestfits_pars
---
`voxel_size_w`, `kd3_sr` and `max_nn` best-fits.
```

```Python
# load bestfit results
for key, val in trees.items():

    keep = (val) & (leaves)
    df_ = df[['x', 'y', 'z']][keep]
    points = loads.DF2array(df_)

    bestfit_file = os.path.join(_data, mockname, 'lia', 'bestfit_%s.npy' %(key))
    res = np.load(bestfit_file, allow_pickle=True)
    res = res.tolist()

    text = 'leaf area=%.2f \n %s=%.4f \n %s=%.4f \n %s=%.4f ' %(res['leafsize'], 
                                                    'voxel_size_w', res['voxel_size_w_bestfit'],
                                                    'kd3_sr', res['kd3_sr_bestfit'],'max_nn', 
                                                    res['max_nn_bestfit'])

    chi2 = lia.leaf_angle(points, mockname, key, res['voxel_size_w_bestfit'], 
                            res['kd3_sr_bestfit'], res['max_nn_bestfit'], save=True,
                                savefig=True, text=text)
```


```{figure} ../figs/leaf_angle_dist_tree_0.png
---
width: 40em
name: lia_dist
---
LIA ($\theta_{L}$) distribution with `weights correction`.
```


```{figure} ../figs/leaf_angle_dist_height_tree_0.png
---
width: 40em
name: lia_dist_h
---
LPC distribution accross different heights in terms of voxels $k$. Bottom - If `Truth` LIA available, $\theta_{L}^{truth} - \theta_{L}$. The different curves show this for different heights ($k$).
```