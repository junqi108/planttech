---
substitutions:
  key1: |
    ```{figure} ../figs/ray.png
    ---
    width: 18em
    name: ray_sample
    ---
    Escheme of ray passing through the `plant region` which corresponds to the red parallelepiped. The red point shows the ray's departure and the green point the ray's end.
    ```
  key2: |
    ```{figure} ../gifs/ray.gif
    ---
    width: 18em
    name: ray_sample_gif
    ---
    Zoom in of ray trajectory scheme. The green cubes as the voxels hitted by the ray. Voxel size is $0.2$.
    ```
---
# `Leaf Area Density` (LAD) estimation

## Intro

Most functions used in this chapter are in library:

``` Python
import lad
```

The LAD method implemented here uses the `Voxel 3D contact-bases frecuency` method first introduced by `HOSOI AND OMASA: VOXEL-BASED 3-D MODELING OF INDIVIDUAL TREES FOR ESTIMATING LAD`.

The model looks like:

```{math}
:label:
LAD(h, \Delta H) = \frac{1}{\Delta H} \sum_{k=m_{h}}^{m_{h}+\Delta H} l(k),
```

where,

```{math}
:label:
l(k) = \alpha(\theta)N(k) \\
    = \alpha(\theta) \cdot \frac{n_{I}(k)}{n_{I}(k) + n_{P}(k)}.
```

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
The `downsample` step is very important as we notice that there's a relation between `downsample` and the `voxel_size` that does not seem to affect the estimation of the LAD. This relationship can be seen in {numref}`downsample_samples` where the lower the `downsample` is, the higher the `voxel_size` needs to be.

```{figure} ../figs/lads_downsample_samples.png
---
width: 40em
name: downsample_samples
---
LAD estimation (blue and red) vs Truth (black) as a function of height. From left to right figure shows estimated LADs for a `downsample` of $75 \%$, $50 \%$, $25 \%$, and $10 \%$ respectively. All estimations use a `voxel_size` of $0.1$.ret0
```

```{admonition} To-Do
:class: important
Results on LAD are dependant of the `voxel_size` and hence, the point cloud density. Currently we rely on the *Truth LAD* to find the correct `voxel_size`. For a real scenario we would like to find the correct value of `voxel_size` based on the PC density or any other intrinsic parameter of the point cloud.
```

(sec:np)=
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

|          |          |
| -------- | -------- |
| {{key1}} | {{key2}} |

(sec:ni)=
## Computing $n_{I}(k)$

The $n_{I}$ per voxel is computed in function `lad.compute_attributes()`. It essentialy voxelize the LPC to get the PR dimensions (which have to be the same as in $n_{P}$). Then, for a numpy boolean 3D-array with voxelize PR dimensions, we fill it with True if there's a point in the voxel.

This function looks for previous `m3s_<treename>_<voxel_size>.npy` result and get the attributes in the form of the same size numpy 3D-array (`m3att`). The attributes are:

- 1 if any LPC in that voxel
- 2 if any beam pass trhough that  voxel
- 3 if none of previous

It requires 4 input parameters `pointsPR`, `resdir`, `voxel_size`, `treename` which were defined in section {ref}`sec:np`. It returns the attributes numpy 3D-array.

## Computing $\alpha(\theta)$

$\alpha(\theta)$ is expressed in terms of $G(\theta)$, 

```{math}
:label:
\alpha(\theta) = \frac{\cos(\theta)}{G(\theta)},
```

where $G(\theta)$ is the mean projection of a unit leaf area on a plane perpendicular to the direction of the laser beam. This quantity is determined with the assumption that leaves are positioned symmetrically with respect to the azimuth anc can be represented as:

```{math}
:label:
G(\theta) =  \sum_{q=1}^{T_{q}} g(q) S(\theta, \theta_{L}(q))
```

where $S(\theta, \theta_{L}(q))$ is expresed in terms of the leaf inclination angle (LIA) $\theta_{L}$ (the zenith angle of the normal to the leaf surface), and $\theta$ is the laser-beam incident zenith angle:

```{math}
:label:
S(\theta, \theta_{L}) = \cos\theta \cos \theta_{L}, \hspace{.5cm} \textrm{for } \theta \leq \pi/2 - \theta_{L}
```

```{math}
:label:
S(\theta, \theta_{L}) = \cos\theta \cos \theta_{L} \left[ 1 + \frac{2}{\pi}(\tan x - x) \right], \hspace{.5cm} \textrm{for } \theta \gt \pi/2 - \theta_{L}
```

```{math}
:label:
x = \cos^{-1}\left( \cot \theta \cot \theta_{L} \right).
```

Here $q$ is the leaf-inclination-angle class and Tq is the total number of leaf-inclination-angle classes. Thus, if there are $18$ leaf-inclination-angle classes from $0◦$ to $90◦$ ($Tq = 18$), then each class consists of a $5◦$ interval. For example, $q = 1$, $q = 9$, and $q = 16$ include the angles from $0◦$ to $4◦$, $40◦$ to $44◦$, and $75◦$ to $79◦$, respectively. $g(q)$ is the distribution of the leaf-inclination-angle class $q$, which is a ratio of the leaf area belonging to class $q$ to total leaf area; $θ_{L}(q)$ is the midpoint angle of class $q$, which is the leaf-inclination angle used to represent class $q$.

This process is done trhough function `lad.Gtheta()`. In function `lad.alpha_k()` we compute $\alpha(\theta)$ for the median of $\theta$, the Beam Inclination Angles (BIA) with respect to zenith, in the Kth layer. We made use of the files `angles_<treename>.npy` and `weights_<treename>.npy` we store previously in the directory `lia` to get $g(q)$. The function `lad.alpha_k()` create three figures inside the `figures` directory:

- `alphas_<treename>_<voxel_size>.png`
- `bia_<treename>_<voxel_size>.png`
- `bia_per_k_<treename>_<voxel_size>.png`

Examples of this three figures can be found in Figures {numref}`alphasplot`, {numref}`biaplot`, and {numref}`biakplot` for a `downsample` of $5 \%$.

```{figure} ../figs/alphas_tree_0_0.2.png
---
width: 40em
name: alphasplot
---
$\alpha(\theta)$ for $0 < \theta < 90$. The black-dashed line is $\alpha(\theta)$ with $G(\theta)$ using all the weighted LIAs, while the coloured-solid lines are with $G(\theta)$ using the weighted LIAs only in the Kth layer. The yellow shadow shows the BIAs range, and the vertical red-dashed line shows the "magic" angle at $\theta = 57.5$ where $\alpha(\theta)$ is close to a constant value of $1.1$.
```

```{figure} ../figs/bia_tree_0_0.2.png
---
width: 40em
name: biaplot
---
The BIAs distribution pre and after angle range correction in Eq.{eq}`angcorr`.
```

```{figure} ../figs/bia_per_k_tree_0_0.2.png
---
width: 40em
name: biakplot
---
The BIAs distribution in the Kth layer.
```

## Estimate LAD

Now that we have $\alpha(\theta)$ in the Kth layer (i.e. $\alpha(\theta, k)$), we can compute the LAI and therefore the LAD. We do this in function `lad.get_LADS()` which requires 4 input parameters:

- `m3att`: The numpy 3D-array attributes we derive in section {ref}`sec:ni`
- `voxel_size`: Voxel Size.
- `kbins`: $\Delta H$ in lengths if K.
- `alphas_k`: `lad.alpha_k()` function output.

This returns a numpy 2D-array with the height and LAD for the corresponding height with zero being the bottom of the PR.

Finally, with function `figures.plot_lads()` we plot LAD as a function of height for:

1. Using correction of $\alpha(\theta, K)$ taking the median of $\theta$ in the Kth layer.
2. Without $\alpha(\theta, K)$ correction
3. Truth LAD from mesh file.

The piece of code below we show all the above mentioned steps plus other minor steps. The output figure is saved in directory `figures` with name `LAD_<treename>_<voxel_size>.png` and shown in {numref}`ladplot`.

```python
if downsample is not None:
    inds_file = os.path.join(resdir, 'inds.npy')
    inds = np.load(inds_file)
    resdir = os.path.join(_data, mockname, 'lad_%s' %(str(downsample)))
    print('downsample:', downsample)
else:
    inds = np.ones(len(df), dtype=bool)
    resdir = os.path.join(_data, mockname, 'lad')

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
    lias, ws = lad.downsample_lia(mockname, key, inds_)
    voxk = lad.get_voxk(pointsPR, voxel_size)
    bia = lad.get_bia(pointsPR, sensorsPR)
    meshfile = lad.get_meshfile(mockname)

    figext = '%s_%s' %(key, str(voxel_size))
    alphas_k = lad.alpha_k(bia, voxk, lias, ws, resdir, meshfile, figext=figext, 
                            klia=False, use_true_lia=True)

    kmax = m3att.shape[2]
    kbins = int(kmax/15)
    print(kbins)
   
    lads_mid = lad.get_LADS(m3att, voxel_size, kbins, alphas_k[:,6], 1)
    lads_0 = lad.get_LADS(m3att, voxel_size, kbins, alphas_k[:,6]*0+1, 1.0)
    lads_mesh = lad.get_LADS_mesh(meshfile, voxel_size, kbins, kmax)

    lads = {'Truth':lads_mesh, 'Correction Mean':lads_mid, 'No Correction':lads_0}

    savefig = os.path.join(resdir, 'figures','LAD_%s.png' %(figext))
    figures.plot_lads(lads, savefig=savefig)
```

```{figure} ../figs/LAD_tree_0_0.2.png
---
width: 25em
name: ladplot
---
LAD as a function of height for (1) solid-red, (2) solid-blue, and (3) solid-black.
```

(sec:finstruc)=
## final structure

In this chapter we create several files that are listed below.

```
root
└───data
    └───test
        │   s0100000.numpy
        │   s0200000.numpy
        │   ...
        │   mesh.ply
        │   scanner_pos.txt
        │   s0100000.npy
        │   s0200000.npy
        │   ...
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
            │   bestfits_pars_treename>.png
```