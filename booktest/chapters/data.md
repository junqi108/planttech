# Data structure

```
root
│   requirements.yml
│   readme.md  
│
└───data
│   └───test
│       │   s0100000.numpy
│       │   s0200000.numpy
│       │   ...
│       │   mesh.ply
│       │   scanner_pos.txt
│   
└───py
    │   loads.py
    │   lia.py
    │   lad.py
    │   ray.py
    │   figures.py

```

## Blensor output transformation

Most of the functions used in this chapter need:

``` Python
import loads
```

In order to get the `LIA` and hence the `LAD`, we need to segmentated the trees and the leaves.

First, we define the name of the directory where the Blensor output data is, in this particular case we will look for directory `test`. Pipeline will look for this directory inside the `data` directory.

```Python
mockname = 'test'
```

Next, we convert Blensor output `txt` files that have fake `numpy` extension to real `npy`. This is done trhough function `loads.numpy2npy()` as shown below,

```Python
loads.numpy2npy(mockname)
```

```{note}
Transforming to `npy` reduce the size of files, besides is much faster to load than the Blensor `txt` output files.
```

The structure looks like,

```
root
|
└───data
│   └───test
│       │   s0100000.numpy
│       │   s0200000.numpy
│       │   ...
│       │   mesh.ply
│       │   scanner_pos.txt
│       │   s0100000.npy
│       │   s0200000.npy
│       │   ...
```


## Tree and leaves segmentation

Now we create the module to segmentate trees. This will be tuned acordingly for each data set, so below module only works for this particular data set.

```Python
def segtree(df, leaves, show=False):

    trees = {}

    if show:
        plt.figure(figsize=(14, 8))

    # centres
    x, y = [0], [0]
    num = 0
    dx, dy = 5, 5

    for i in x:
        for j in y:
            
            keep = np.ones(len(df['x']), dtype=bool)
            keep &= (df['x'] < i+dx) & (df['x'] > i-dx)
            keep &= (df['y'] < j+dy) & (df['y'] > j-dy)

            trees['tree_%s' %(str(num))] = keep
            
            if show:
                plt.scatter(df['x'][leaves & keep], df['y'][leaves & keep], s=0.5, label=num)
                        
            num += 1

    if show:
        plt.legend()
    
    return trees
```

We segmentate the trees below,

```Python
# load data into a pandas data frame
df = loads.npy2pandas(mockname)
# extract leaves. Boolean array output
leaves = loads.extract_leaves(df, show=False)
# extract trees. Dictionary with boolean arrays output
trees = segtree(df, leaves)
```

That's it! So what we just did? First, with function `loads.npy2pandas()` we load all the `npy` files into a pandas DataFrame (`DF`) and we add three more columns with the $x$, $y$, and $z$ positions of the sensors that are stored in file `scanner_pos.txt`. Then, since this is a mockup dataset, we can easily separate the leaves from everythin else in the point cloud (`PC`). We do this with function `loads.extract_leaves()` that requires the pandas `DF` as input. Finally, we invoke the above module to segmentate trees that requires the pandas `DF` as well.

outputs from this are:

* `df`: Pandas DF with the entire PC
* `leaves`: numpy boolean array of PC dimensions with True for Points concerning leaves only
* `trees`: python dictionary where each entry contains one tree in the form of boolena array with PC dimensions

Below piece of code shows an example of how to visualize the leaves points from one tree only, this will be know as the `Leaves Point Cloud` (LPC) and is shown in Fig. {numref}`lpc`.

```Python
# show the point cloud from leaves of firs tree only
keep = (trees['tree_0']) & (leaves)
loads.showPCfromDF(df[keep])
```

```{figure} ../gifs/lpc.gif
---
width: 30em
name: lpc
---
Leaf Point Cloud.
```

In the subsequent chapters we will be comparing our estimations with the *True* values using the `mesh.ply` file located in the root directory `test`. The following piece of code shows how we can visualize this mesh that requires importing the library `lad`. Fig. {numref}`mesh` show this mesh.

```Python
import lad
meshfile = lad.get_meshfile(mockname)
lad.see_mesh(meshfile)
```

```{figure} ../gifs/mesh.gif
---
width: 30em
name: mesh
---
Mesh from Toy tree.
```