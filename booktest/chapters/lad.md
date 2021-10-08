# `Leaf Area Density` (LAD) estimation

Most functions used in this chapter are in library:

``` Python
import lad
```

The LAD method implemented here uses the `Voxel 3D contact-bases frecuency` method first introduced by `HOSOI AND OMASA: VOXEL-BASED 3-D MODELING OF INDIVIDUAL TREES FOR ESTIMATING LAD`.

The model looks like:

$$LAD(h, \Delta H) = \frac{1}{\Delta H} \sum_{k=m_{h}}^{m_{h}+\Delta H} l(k),$$

where,

$$l(k) = \alpha(\theta)N(k) \\
    = \alpha(\theta) \cdot \frac{n_{I}(k)}{n_{I}(k) + n_{P}(k)}.$$

$l(k)$ is the `Leaf Area Index` (LAI) of the kth horizontal layer of the voxel array within a plant region, $\Delta H$ is the horizontal layer thickness, and $m_{h}$ and $m_{h}+\Delta H$ are the voxel coordinates on the vertical axis equivalent to height $h$ and $h+\Delta H$ in orthogonal coordinates ($h = \Delta k \times m_{h}$). The LAI of the kth horizontal layer $l(k)$ is the product of the contact frequency $N(k)$ of laser beams in the kth layer and the coefficient $\alpha(\theta)$, which corrects for leaf inclination at laser incident zenith angle $\theta$.

$n_{I}(k)$ is the number of voxels where the laser beams is intercepted by the kth layer, $n_{P}(k)$ is the number of voxels where the laser beams passed through the kth layer, and $n_{I}(k) + n_{P}(k)$ is the total number of voxels where the incident laser beams reach the kth layer.

Despite the complexity of this method, it requieres only one parameter, the `voxel_size`. We will introduce a second parameter, the `downsample` whose importance will be explained later. The main steps towards LAD estimation are:

1. Computing $n_{P}(k)$
2. Computing $n_{I}(k)$
3. Computing $\alpha(\theta)$
4. Estimate LAD

For this example, we will usea a `voxel_size` = 0.2 and `downsample` = 0.05 which means that we downsample our whole data to only $5\%$. A reminder that as well as in for the LIA, the following process is per tree.

```Python
downsample = 0.05
voxel_size = 0.2
```

```{important}
The `downsample` step is very important as we notice that there's a relation between `downsample` and the `voxel_size` that does not seem to affect the estimation of the LAD. This relationship can be seen in figure {numref}`downsample_samples` where the lower the `downsample` is, the higher the `voxel_size` needs to be.

```{figure} ../figs/lads_downsample_samples.png
---
width: 40em
name: downsample_samples
---
LAD estimation (blue and red) vs Truth (black) as a function of height. From left to right figure shows estimated LADs for a `downsample` of $75 \%$, $50 \%$, $25 \%$, and $10 \%$ respectively. All estimations use a `voxel_size` of $0.1$.
```

## Computing $n_{P}(k)$

Seeing where the beam pass through in the voxelize `Plant Region` (PR) is a tipycal ray tracing problem and it's reduced to see whether the ray hit or not an axis align bounding box (AABB).

The below module grabs the requiered downsample percentge of the data with a random subsample and if it's the first time we ran this, it will create the directory `lad_<downsmple>`. All the subsequent results will be stored inside this directory. The first time a particular downsample is ran, it will store the file `inds.npy` containing a boolean array with size of the pandas DF where True being the selected random subsample requested. If we change the `voxel_size` but not the `downsample`, then the below module will look first for the `inds.npy` instead of searching for another random subsample, this to maintain uniformity between different voxels sizes approaches.

The function `main` does the magic here, it has to be ran per tree and requires 6 input parameters:

- `points`: $x$, $y$ and $z$ coordinates from the downsample data in the form of numpy array.
- `sensors`: $x$, $y$ and $z$ coordinates of sensor responsible from each point in `points` parameter above.
- `pointsPR`: `points` above filtered to the LPC.
- `voxel_size`: Voxel Size.
- `resdir`: Name of output directory for the specific `downsample`.
- `treename`: Name/index of tree.

```{note}
`pointsPR` is require to get the same voxelization dimensions as in $n_{I}$.
```
This function returns two files:

- `m3s_<treename>_<voxel_size>.npy`: numpy boolean 3D-array with number of voxels dimensions. True if a beam hit the voxel.
- `m3count_<treename>_<voxel_size>.npy`: numpy 3D-array with number of voxels dimensions. Each entry contains the number of beams that passed trhough that voxel.

```{admonition} To-Do
:class: important
Note that this is the slowest module of the entire pipeline, taking up to 5 minutes for a sample of 10,000 beams. This can be improved easlily if binding with a `C++` ray AABB module instead.
```

Below we show the piece of code that computes this,

```Python

POINTS = loads.DF2array(df[['x', 'y', 'z']])
SENSORS = loads.DF2array(df[['sx', 'sy', 'sz']])

if downsample is not None:

    resdir = os.path.join(_data, mockname, 'lad_%s' %(str(downsample)))
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

    resdir = os.path.join(_data, mockname, 'lad')
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
