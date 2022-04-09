import numpy as np
import os, sys
import matplotlib.pyplot as plt

import pyrr
import itertools

# basedir = os.path.dirname(os.getcwd())
basedir = os.path.abspath(os.path.join(os.getcwd() ,"../"))
_py = os.path.join(basedir, 'py')
_data = os.path.join(basedir, 'data')

sys.path.insert(1, _py)
import loads
import lia
import ray as rayt
import lad
import figures

import warnings
warnings.filterwarnings("ignore")

def segtree(df, leaves, show=False):

    trees = {}

    if show:
        plt.figure(figsize=(14, 8))

    # centres
    x, y = [0], [0]
    num = 0
    dx, dy = 2, 2
    # dx, dy = 5, 5

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

def voxel_subsampling(voxel_size, POINTS):

    nb_vox = np.ceil((np.max(POINTS, axis=0) - np.min(POINTS, axis=0))/voxel_size)
    ni, nj, nk = nb_vox
    print('min point:', np.min(POINTS, axis=0))
    print('max point:', np.max(POINTS, axis=0))
    print('Number of voxels: i:%d, j:%d, k:%d --> Total: %d' %(ni, nj, nk, np.product(nb_vox)))

    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((POINTS - np.min(POINTS, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)
    print('Number of non-empty voxels: %d' %(len(non_empty_voxel_keys)))

    voxel_grid={}
    voxel_grid_ptsidx = {}
    grid_barycenter,grid_candidate_center = [], []
    last_seen=0

    for idx, vox in enumerate(non_empty_voxel_keys):

        idxs_per_vox = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
        voxel_grid[tuple(vox)] = POINTS[idxs_per_vox]
        voxel_grid_ptsidx[tuple(vox)] = idxs_per_vox

        # grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))

        idx_grid_candidate_center = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
        grid_candidate_center.append(voxel_grid_ptsidx[tuple(vox)][idx_grid_candidate_center])

        last_seen+=nb_pts_per_voxel[idx]

    # print('Downsampling percentage: %.1f %%' %(100 * len(grid_candidate_center) / len(POINTS)))
    # minpoint = np.min(POINTS, axis=0)

    return list(grid_candidate_center) #, minpoint

def random_downsample(mockname, N, downsample):

    resdir = os.path.join(_data, mockname, 'random_%s' %(str(downsample)))
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    outdir = os.path.join(resdir, 'inds.npy')
    if os.path.exists(outdir):
        print('inds file already exists for donwnsample of %.3f at %s' %(downsample, outdir))

        idx = np.load(outdir)

    else:

        print('inds not been created yet for donwnsample of %.3f' %(downsample))
        idx = np.random.randint(0, N, int(N * downsample))
        # inds = np.zeros(N, dtype=bool)
        # inds[idx] = True

        np.save(outdir, idx)

    return idx


if __name__ == "__main__":

    mockname = 'test'
    # voxel_size = 0.15
    downsample = None
    # downsample = 0.3

    for voxel_size in [0.05, 0.1, 0.15]:

        print('========= DS:%s, VS:%s =========' %(str(downsample), str(voxel_size)))

        # load data into a pandas data frame
        df = loads.npy2pandas(mockname)
        N = len(df)
        print(N)

        if downsample is not None:
            inds = random_downsample(mockname, N, downsample)
            print('Random downsampling...')
        else:
            inds = voxel_subsampling(voxel_size, df[['x', 'y', 'z']].to_numpy())
            print('Voxel downsampling...')

        print('Downsampling percentage: %.1f %%' %(100 *  len(inds) / len(df['x'])))

        df = df.iloc[inds]
        POINTS = df[['x', 'y', 'z']].to_numpy()
        SENSORS = df[['sx', 'sy', 'sz']].to_numpy()

        # Compute lower point
        minpoint = np.min(POINTS, axis=0)
        print('minpoint:', minpoint)


        # extract leaves. Boolean array output
        leaves = loads.extract_leaves(df, show=False)
        # extract trees. Dictionary with boolean arrays output
        trees = segtree(df, leaves, show=False)

        inPR = (leaves) & (trees['tree_0'])
        minBB, maxBB = np.min(POINTS[inPR.values], axis=0), np.max(POINTS[inPR.values], axis=0)

        # Make sure Plant Region min & max points are multiples of voxel size
        # to match first voxelization where we implemented the downsampling
        minpointPR = minpoint + np.floor(np.abs(minpoint - minBB)/voxel_size) * voxel_size
        maxpointPR = minpoint + np.ceil(np.abs(minpoint - maxBB)/voxel_size) * voxel_size
        boxPR = pyrr.aabb.create_from_bounds(minpointPR, maxpointPR)

        lines = np.stack((POINTS, SENSORS), axis=1)
        f = lambda line: pyrr.geometric_tests.ray_intersect_aabb(pyrr.ray.create_from_line(line), boxPR) is not None
        res = np.array(list(map(f, lines)))

        POINTS, SENSORS = POINTS[res], SENSORS[res]

        leaves = leaves[res]

        for key, val in trees.items():
            trees[key] = val[res]

        # save indexes of voxel-based downsample

        idxs = np.array(inds)[res]

        if downsample is not None:
            dirname = 'random_%s' %(str(downsample))
            resdir = os.path.join(_data, mockname, dirname, 'lad_%s' %(str(voxel_size)))
        else:
            dirname = 'voxel'
            resdir = os.path.join(_data, mockname, dirname, 'lad_%s' %(str(voxel_size)))

        if not os.path.exists(resdir): os.makedirs(resdir)
        outdir = os.path.join(resdir, 'inds.npy')
        np.save(outdir, idxs)

        # Ray tracing
        sample = None
        inPR = (leaves) & (trees['tree_0'])

        if sample is not None:
            print('# iter...', len(POINTS[::sample]))
            m3s = rayt.main2(POINTS[::sample], SENSORS[::sample], POINTS[inPR], voxel_size, resdir, 'tree_0', (minpointPR, maxpointPR), show=True)
        else:
            print('# iter...', len(POINTS))
            print('Results will be saved at %s' %(resdir))
            print('-------------')
            m3s = rayt.main2(POINTS, SENSORS, POINTS[inPR], voxel_size, resdir, 'tree_0', (minpointPR, maxpointPR), show=False)


