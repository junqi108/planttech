---
substitutions:
  klad_0_05: |
    ```{figure} ../figs/kiwi_LAD_tree_0_0.05.png
    ---
    width: 18em
    ---
    ```
  klad_0_1: |
    ```{figure} ../figs/kiwi_LAD_tree_0_0.1.png
    ---
    width: 18em
    ---
    ```
  klad_0_15: |
    ```{figure} ../figs/kiwi_LAD_tree_0_0.15.png
    ---
    width: 18em
    ---
    ```
  klad_1_05: |
    ```{figure} ../figs/kiwi_LAD_tree_1_0.05.png
    ---
    width: 18em
    ---
    ```
  klad_1_1: |
    ```{figure} ../figs/kiwi_LAD_tree_1_0.1.png
    ---
    width: 18em
    ---
    ```
  klad_1_15: |
    ```{figure} ../figs/kiwi_LAD_tree_1_0.15.png
    ---
    width: 18em
    ---
    ```
  klad_2_05: |
    ```{figure} ../figs/kiwi_LAD_tree_2_0.05.png
    ---
    width: 18em
    ---
    ```
  klad_2_1: |
    ```{figure} ../figs/kiwi_LAD_tree_2_0.1.png
    ---
    width: 18em
    ---
    ```
  klad_2_15: |
    ```{figure} ../figs/kiwi_LAD_tree_2_0.15.png
    ---
    width: 18em
    ---
    ```
  klad_3_05: |
    ```{figure} ../figs/kiwi_LAD_tree_3_0.05.png
    ---
    width: 18em
    ---
    ```
  klad_3_1: |
    ```{figure} ../figs/kiwi_LAD_tree_3_0.1.png
    ---
    width: 18em
    ---
    ```
  klad_3_15: |
    ```{figure} ../figs/kiwi_LAD_tree_3_0.15.png
    ---
    width: 18em
    ---
    ```
  klad_4_05: |
    ```{figure} ../figs/kiwi_LAD_tree_4_0.05.png
    ---
    width: 18em
    ---
    ```
  klad_4_1: |
    ```{figure} ../figs/kiwi_LAD_tree_4_0.1.png
    ---
    width: 18em
    ---
    ```
  klad_4_15: |
    ```{figure} ../figs/kiwi_LAD_tree_4_0.15.png
    ---
    width: 18em
    ---
    ```
---

# `Leaf Area Density` (LAD) estimation

## Intro

We have the same file structure as in {ref}`sec:finstruc`.

```
root
│   requirements.yml
│   readme.md  
│
└───data
    └───kiwifruit
        │   lidar.laz
        │   trajectory.dbf
        └───lad_<downsample>
        │   │   inds.npy
        │   │   m3s_<treename>_<voxel_size>.npy
        │   │   m3count_<treename>_<voxel_size>.npy
        │   └───figures
        │       │   alphas_<treename>_<voxel_size>.png
        │       │   bia_<treename>_<voxel_size>.png
        │       │   bia_per_k_<treename>_<voxel_size>.png
        │       │   LAD_<treename>_<voxel_size>.png
        └───lia
            │   angles_<treename>.npy
            │   weights_<treename>.npy
            │   leaf_angle_dist_<treename>.png
            │   leaf_angle_dist_height_<treename>.png
```

## LAD results for three different `voxel_size` inputs

Since the relation between the two input parameters (`downsample`, and `voxel_size`) of this pipeline to derive the LAD is left for another study, we choose three different `voxel_size` inputs $0.05$, $0.1$, and $0.15$ which corresponds to $5$, $10$, and $15$ cm respectively. We choose one `downsample` value of $0.005$ which corresponds to $0.5$ per cent out of total LiDAR beams of $\sim 40,000,000$.

The code to compute $n_{P}(k)$ is similar to the one in {ref}`sec:np` with some minor differences,

```python
POINTS = loads.DF2array(df[['x', 'y', 'z']])
SENSORS = loads.DF2array(df[['xs', 'ys', 'zs']])

if downsample is not None:

    resdir = os.path.join(_data, name, 'lad_%s' %(str(downsample)))
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    outdir = os.path.join(resdir, 'inds.npy')
    if os.path.exists(outdir):
        print('inds file already exists for donwnsample of %.3f at %s' %(downsample, outdir))

        inds = np.load(outdir)

        points = POINTS[inds]
        sensors = SENSORS[inds]

    else:

        print('inds not been created yet for donwnsample of %.3f' %(downsample))
        idx = np.random.randint(0, len(df), int(len(df) * downsample))
        inds = np.zeros(len(df), dtype=bool)
        inds[idx] = True

        points = POINTS[inds]
        sensors = SENSORS[inds]

        np.save(outdir, inds)

else:

    resdir = os.path.join(_data, name, 'lad')
    if not os.path.exists(resdir):
        os.makedirs(resdir)

if sample is not None:

    idx = np.random.randint(0, len(df), int(sample))
    points = POINTS[idx]
    sensors = SENSORS[idx]

for key, val in trees.items():

    inPR = (val) & (leaves) & (inds)
    pointsPR = POINTS[inPR]
    m3s, m3count= rayt.main(points, sensors, pointsPR, voxel_size, resdir, key, show=show)
```

The code to compute $n_{I}(k)$ and LAD is as well similar to {ref}`sec:ni`

```python
if downsample is not None:
    inds_file = os.path.join(resdir, 'inds.npy')
    inds = np.load(inds_file)
    resdir = os.path.join(_data, name, 'lad_%s' %(str(downsample)))
    print('downsample:', downsample)
else:
    inds = np.ones(len(df), dtype=bool)
    resdir = os.path.join(_data, name, 'lad')

isfigures = os.path.join(resdir, 'figures')
if not os.path.exists(isfigures):
    os.makedirs(isfigures)

print('voxel_size:', voxel_size)

for key, val in trees.items():

    inPR = (val) & (leaves) & (inds)
    pointsPR = POINTS[inPR]
    sensorsPR = SENSORS[inPR]

    m3att = lad.compute_attributes(pointsPR, resdir, voxel_size, key)
    # get in down sample boolean array for LPC size
    inds_ = inds[(val) & (leaves)]
    lias, ws = lad.downsample_lia(name, key, inds_)
    voxk = lad.get_voxk(pointsPR, voxel_size)
    bia = lad.get_bia(pointsPR, sensorsPR)

    figext = '%s_%s' %(key, str(voxel_size))
    alphas_k = lad.alpha_k(bia, voxk, lias, ws, resdir,    figext=figext, klia=False, use_true_lia=False)

    kbins = 1
    
    lads_mid = lad.get_LADS(m3att, voxel_size, kbins, alphas_k[:,6], 1)
    lads_0 = lad.get_LADS(m3att, voxel_size, kbins, alphas_k[:,6]*0+1, 1.0)

    lads = {'Correction Mean':lads_mid, 'No Correction':lads_0}

    savefig = os.path.join(resdir, 'figures','LAD_%s.png' %(figext))
    figures.plot_lads(lads, savefig=savefig)
```

### $\alpha(\theta)$ results

In {numref}`kalpha_05`, {numref}`kalpha_1`, and {numref}`kalpha_15` we show $\alpha(\theta)$ for `tree_1` and for the three different `voxel_size` values.


```{figure} ../figs/kiwi_alphas_tree_1_0.05.png
---
width: 30em
name: kalpha_05
---
$\alpha(\theta)$ for `tree_1` using a `voxel_size` of $0.05$.
```

```{figure} ../figs/kiwi_alphas_tree_1_0.1.png
---
width: 30em
name: kalpha_1
---
$\alpha(\theta)$ for `tree_1` using a `voxel_size` of $0.1$.
```

```{figure} ../figs/kiwi_alphas_tree_1_0.15.png
---
width: 30em
name: kalpha_15
---
$\alpha(\theta)$ for `tree_1` using a `voxel_size` of $0.15$.
```

LAD results for `tree_0` trhough `tree_4`, and for each `voxel_size` is shown in Figures below.

### LAD for `tree_0`

| `voxel_size=0.05`|`voxel_size=0.1`|`voxel_size=0.15`|
| ---------------- | -------------- | --------------- |
|   {{klad_0_05}}  |  {{klad_0_1}}  |   {{klad_0_15}} |

### LAD for `tree_1`

| `voxel_size=0.05`|`voxel_size=0.1`|`voxel_size=0.15`|
| ---------------- | -------------- | --------------- |
|   {{klad_1_05}}  |  {{klad_1_1}}  |   {{klad_1_15}} |

### LAD for `tree_2`

| `voxel_size=0.05`|`voxel_size=0.1`|`voxel_size=0.15`|
| ---------------- | -------------- | --------------- |
|   {{klad_2_05}}  |  {{klad_2_1}}  |   {{klad_2_15}} |


### LAD for `tree_3`

| `voxel_size=0.05`|`voxel_size=0.1`|`voxel_size=0.15`|
| ---------------- | -------------- | --------------- |
|   {{klad_3_05}}  |  {{klad_3_1}}  |   {{klad_3_15}} |

### LAD for `tree_4`

| `voxel_size=0.05`|`voxel_size=0.1`|`voxel_size=0.15`|
| ---------------- | -------------- | --------------- |
|   {{klad_4_05}}  |  {{klad_4_1}}  |   {{klad_4_15}} |