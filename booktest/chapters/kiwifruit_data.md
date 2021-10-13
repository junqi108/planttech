# Kiwifruit MLS LiDAR data structure

```
root
│   requirements.yml
│   readme.md  
│
└───data
│   └───kiwifruit
│       │   lidar.laz
│       │   trajectory.dbf
│   
└───py
    │   loads.py
    │   lia.py
    │   lad.py
    │   ray.py
    │   figures.py

```

The data we will analize here corresponds to a kiwifruit Orchard and LiDAR was taken using a **Hovermap Mobile Laser Scanner** (MLS). LiDAR of kiwifruit contains around $40,000,000$ laser beams. There is two input files needed, first is the LiDAR data file which is called `lidar.laz`, and second, `trajectory.dbf` which contains the coordinates of the trajectory of the MLS. These, and all the subsequent files created by pipeline, will be placed at `kiwifruit` directory under `data` directory.

As usual, the modules needed are:

```python
import loads
import lia
import ray as rayt
import lad
import figures
```

We start defining the name of the directory where the LiDAR data is,

```python
name = 'kiwifruit'
```

the input files `lidar.laz` and `trajectory.dbf` are load with,

```python
# load files
las = loads.loadlaz(name)
traj = loads.loaddbf(name)
```

These point clouds can be seen in {numref}`kiwipc`.

```{figure} ../gifs/kiwifruit_pc.gif
---
width: 30em
name: kiwipc
---
3D vizualization of the kiwifruit point cloud (gray dots) and the trajectory of the MLS (red dots).
```

## Interpolation of trajectory

The trajectory file `trajectory.dbf` has been downsampled to $1$ every $50$ points, therefore we interpolate in order to get each beams origin coordinate used for ray tracing in a later stage. We do this with,

```python
df = loads.coordsDF(las, traj)
```

that tooks the LiDAR and trajectory data and generates a Pandas DF with orgin beams coordinates $xs$, $ys$, and $zs$, and the beam en point coordinates $x$, $y$, and $z$.

The downsampled trajectory of MLS can be seen in {numref}`traj`.

```{figure} ../figs/K04A_01_traj.png
---
width: 30em
name: traj
---
Aerial view of MLS trajectry for the current LiDAR data set.
```

## Leaves segmentation

First we choose a region where we believe the data is most complete and kiwifruits live with no other structure. This region is shown in {numref}`aerial_sel` with dimensions $50 \times 10 = 500$ $m^2$. The rectangle has the following linear equations delimiters,

```{math}
:label: region
y > 0.34x - 4 \\
y < 0.34x + 6 \\
x < -0.35y - 10 \\
x > -0.35y - 60
```

```{figure} ../figs/aerial_selected_zoom.png
---
width: 30em
name: aerial_sel
---
Aerial view of LiDAR. Red rectangle shows the area choosen for the analisis.
```

Now that we have choose the region to work with, let's continue with the leave segmentation. We perform the leave segmentation by taking height gap using the $z$ coordinate,

```{math}
:label: zlimit
z > 1.2 \\
z < 3.8
```
Next, within this gap, remove outliers by taking bins in the $x$ coordinate and take the $1$ and $99$ per cent data in $z$ coordinate for that bin. This process is shown in {numref}`foliage`,


```{figure} ../gifs/side_selected_foliage.gif
---
width: 30em
name: foliage
---
Side view of LiDAR selected region in red and the foliage selection in colours using the percentile method.
```

The piece of code that implements Eq.{eq}`region`, Eq.{eq}`zlimit` and the outlier removal is shown below,

```python
# select the region
keep = (df['y'] > 0.34*df['x'] - 4.0) & (df['y'] < 0.34*df['x'] + 6)
keep &= (df['x'] < -0.35*df['y'] - 10) & (df['x'] > -0.35*df['y'] - 60)

# Set the foliage min and max height
keepz = (df['z'] > 1.2) & (df['z'] < 3.8)

# remove outliers using percentiles
res, mask = loads.remove_outliers(df['x'][keep & keepz], df['z'][keep & keepz])
leaves = np.zeros(len(df['x']), dtype=bool)
leaves[(keep) & (keepz)] = mask
```

## Tree segmentation

