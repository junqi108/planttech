# `Leaf Inclination Angle` (LIA) estimation

## Intro

We have the same file structure as in {ref}`sec:lia` but without the files that rely on the mesh data as we don't have any here.

```
root
│   requirements.yml
│   readme.md  
│
└───data
    └───kiwifruit
        │   lidar.laz
        │   trajectory.dbf
        └───lia
            │   angles_<treename>.npy
            │   weights_<treename>.npy
            │   leaf_angle_dist_<treename>.png
            │   leaf_angle_dist_height_<treename>.png
```

The code that computes the weighted LIA per tree is shown below,

```python
for key, val in trees.items():

    keep = (val) & (leaves)
    df_ = df[['x', 'y', 'z']][keep]
    points = loads.DF2array(df_)

    voxel_size_w = 0.01
    kd3_sr = 0.1
    max_nn = 15

    text = '%s=%.4f \n %s=%.4f \n %s=%.4f ' %(
                            'voxel_size_w', voxel_size_w,
                            'kd3_sr', kd3_sr,
                            'max_nn', max_nn)

    lia.leaf_angle(points, name, key, voxel_size_w, kd3_sr, max_nn, save=True,
                                savefig=True, text=text, voxel_size_h=0.1, ismock=False,
                                ylim=0.017, ylimh=0.35)
```

Note that we have used the following input parameters in `lia.leaf_angle`,

```python
voxel_size_w = 0.01
kd3_sr = 0.1
max_nn = 15
```

Results of the LIA for all trees are shown in {numref}`klia0`, {numref}`klia1`, {numref}`klia2`, {numref}`klia3`, and {numref}`klia4`.

```{figure} ../figs/kiwi_leaf_angle_dist_tree_0.png
---
width: 30em
name: klia0
---
LIA ($\theta_{L}$) distribution with `weights correction` of `tree_0`.
```

```{figure} ../figs/kiwi_leaf_angle_dist_tree_1.png
---
width: 30em
name: klia1
---
LIA ($\theta_{L}$) distribution with `weights correction` of `tree_1`.
```

```{figure} ../figs/kiwi_leaf_angle_dist_tree_2.png
---
width: 30em
name: klia2
---
LIA ($\theta_{L}$) distribution with `weights correction` of `tree_2`.
```

```{figure} ../figs/kiwi_leaf_angle_dist_tree_3.png
---
width: 30em
name: klia3
---
LIA ($\theta_{L}$) distribution with `weights correction` of `tree_3`.
```

```{figure} ../figs/kiwi_leaf_angle_dist_tree_4.png
---
width: 30em
name: klia4
---
LIA ($\theta_{L}$) distribution with `weights correction` of `tree_4`.
```