
# Toy tree and sensor specifications

## Dimensions

The toy tree we chose has some complexity as shown in {numref}`toy_tree`. It has a `leaf size` of $0.25$ and dimensions in meters (m):

- $\Delta x = 9.1$
- $\Delta y = 9.2$
- $\Delta z = 9.6$

```{figure} ../figs/toy_tree.png
---
width: 40em
name: toy_tree
---
Toy tree leafless (left) and with leafs (right).
```

## Scann pattern and sensor resolution

The toy tree was scanned $7$ times as the {numref}`scanntree` shows. Each scan has a resolution of $150$ points across vertical and horizontal sensor, and a field of view of $40$ degrees for each side. Total number of points per sensor scan is: $150 \times 150 = 22,500$.


```{figure} ../gifs/scann.gif
---
width: 30em
name: scanntree
---
Toy tree scene with sensors scanner patter.
```