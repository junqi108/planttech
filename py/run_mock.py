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

    print('Downsampling percentage: %.1f %%' %(100 * len(grid_candidate_center) / len(POINTS)))
    minpoint = np.min(POINTS, axis=0)

    return list(grid_candidate_center), minpoint

def runall(pointsPR, sensorsPR, voxel_size, tree, N, PRbounds, kbins=None):

    resdir = os.path.join(_data, mockname, 'lad_%s' %(str(voxel_size)))
    inds_file = os.path.join(resdir, 'inds.npy')
    inds = np.load(inds_file)

    resdir = os.path.join(_data, mockname, 'lad_%s' %(str(voxel_size)))

    inds_lia = np.load(os.path.join(_data, mockname, 'lia', 'inds.npy'))

    isfigures = os.path.join(resdir, 'figures')
    if not os.path.exists(isfigures):
        os.makedirs(isfigures)

    attributes2_file = os.path.join(resdir, 'm3s_%s_%s.npy' %(tree, str(voxel_size)))
    if os.path.isfile(attributes2_file):
        m3b = np.load(attributes2_file)

    print('voxel_size:', voxel_size)

    m3att = lad.compute_attributes(pointsPR, resdir, voxel_size, tree, PRbounds)

    mask1 = np.zeros(N, bool)
    mask2 = mask1.copy()

    mask1[inds] = True
    mask2[inds_lia] = True

    lias, ws = lad.downsample_lia(mockname, tree, np.where(mask1[mask2])[0])
    voxk = lad.get_voxk(pointsPR, PRbounds, voxel_size)
    bia = lad.get_bia(pointsPR, sensorsPR)
    meshfile = lad.get_meshfile(mockname)

    figext = '%s_%s' %(tree, str(voxel_size))
    
    alphas_k = lad.alpha_k(bia, voxk, lias, ws, resdir, meshfile, figext=figext, 
                            klia=False, use_true_lia=True)

    kmax = m3b.shape[2]
    if kbins is None:
        kbins = int(kmax/15)
    print(kbins)

    lads_mid, clai = lad.get_LADS2(pointsPR, kmax, voxel_size, kbins, alphas_k[:,6], PRbounds, tree, resdir)
    lads_0, _ = lad.get_LADS2(pointsPR, kmax, voxel_size, kbins, alphas_k[:,6]*0+1, PRbounds, tree, resdir)
    # lads_mid_old, _ = lad.get_LADS2(pointsPR, kmax, voxel_size, kbins, alphas_k[:,6], PRbounds, tree, resdir, oldlad=True)
    lads_mid_old = lad.get_LADS(m3att, voxel_size, kbins, alphas_k[:,6], alpha2=1)
    lads_mesh = lad.get_LADS_mesh(meshfile, voxel_size, kbins, kmax, PRbounds)

    # lads = {'Truth':lads_mesh, 'Correction Mean':lads_mid, 'No Correction':lads_0, 'Correction Weights':lads_mid_w}#, 'Correction counts':lads_mid_counts}
    lads = {'Truth':lads_mesh, 'Correction Mean':lads_mid, 'No Correction':lads_0, 'Correction Mean OLD':lads_mid_old}
    # clai = lad.get_clai(m3att, alphas_k)
    attributes_file = os.path.join(resdir, 'm3s_%s_%s.npy' %(tree, str(voxel_size)))
    if os.path.isfile(attributes_file):
        RT = 'Y'
    else:
        RT = 'N'
        
    text = {'tree':tree, 'VS':voxel_size, 'RT':RT, 'CLAI':np.round(clai, 3)}
    txt = []
    for key, val in text.items():
        txt.append('%s=%s \n' %(key, str(val)))
    text = (' ').join(txt)

    savefig = os.path.join(resdir, 'figures','LAD_%s.png' %(figext))
    figures.plot_lads(lads, text, savefig=savefig)


if __name__ == "__main__":

    mockname = 'kiwi'
    voxel_size = 0.15
    kbins = 1
    print('Voxel Size=', voxel_size)

    # load data into a pandas data frame
    df = loads.npy2pandas(mockname)
    N = len(df)


    # Implement and keep Downsamplied points
    inds, minpoint = voxel_subsampling(voxel_size, df[['x', 'y', 'z']].to_numpy())


    df = df.iloc[inds]
    POINTS = df[['x', 'y', 'z']].to_numpy()
    SENSORS = df[['sx', 'sy', 'sz']].to_numpy()


    # extract leaves. Boolean array output
    leaves = loads.extract_leaves(df, show=False)
    # extract trees. Dictionary with boolean arrays output
    trees = segtree(df, leaves, show=False)


    # econd downsampling: keep only points that colide with Plant Region
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

    resdir = os.path.join(_data, mockname, 'lad_%s' %(str(voxel_size)))
    if not os.path.exists(resdir): os.makedirs(resdir)
    outdir = os.path.join(resdir, 'inds.npy')
    np.save(outdir, idxs)

    print('Starting Ray tracing...')
    print('# of rays:', len(POINTS))

    sample = None
    print('# iter...', len(POINTS))

    inPR = (leaves) & (trees['tree_0'])

    if sample is not None:
        m3s = rayt.main2(POINTS[::sample], SENSORS[::sample], POINTS[inPR], voxel_size, resdir, 'tree_0', (minpointPR, maxpointPR), show=True)
    else:
        m3s = rayt.main2(POINTS, SENSORS, POINTS[inPR], voxel_size, resdir, 'tree_0', (minpointPR, maxpointPR), show=False)
        
    print('Ray tracing done...')
    
    runall(POINTS[inPR], SENSORS[inPR], voxel_size, 'tree_0', N, (minpointPR, maxpointPR), kbins=kbins)