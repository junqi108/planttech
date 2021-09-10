
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys, glob
import laspy as lp
from scipy.stats import chisquare
from time import process_time

import figures

__author__ = 'Omar A. Ruiz Macias'
__copyright__ = 'Copyright 2021, PLANTTECH'
__version__ = '0.1.0'
__maintainer__ = 'Omar A. Ruiz Macias'
__email__ = 'omar.ruiz.macias@gmail.com'
__status__ = 'Dev'

# Global
basedir = os.path.dirname(os.getcwd())
_py = os.path.join(basedir, 'py')
_data = os.path.join(basedir, 'data')
_images = os.path.join(basedir, 'images')


def beam_direction(yaw, pitch):
    '''
    Get laser beam direction from YAW and PITCH
    '''
    
    x = np.cos(yaw)*np.cos(pitch)
    y = np.sin(yaw)*np.cos(pitch)
    z = np.sin(pitch)
    
    points = np.vstack((np.array(x), np.array(y), np.array(z))).transpose()
    
    return points

def trace_beam(points, beam_angles, distance, res=0.1):

    tracers = {'p':[], 'pb':[], 'linepts':[]}

#     beam_angles = np.vstack((np.array(df['yaw']), np.array(df['pitch']))).transpose()
#     dist = np.array(df['distance'])

    # unitary vector with direction of beam from camera point of view
    uv = beam_direction(beam_angles.T[0], beam_angles.T[1]) 

    for i in range(uv.shape[0]):
        
        # proyect unitary vector to actual distance of beam and invert it to point cloud data view
        N = int(distance[i]/res)
        pb = -1 * uv[i] * np.linspace(0.01, distance[i], N)[:, np.newaxis] # this traces the beam with N points

        # Moving proyected unitary vector (i.e. the beam) to its reall position
        linepts = points[i] + pb

        # create matrix with position of the origin beam and position of the intersected beam
#         twopoints = np.vstack([p[i], linepts[-1]])

        tracers['p'].append(points[i])
        tracers['pb'].append(linepts[-1])
        tracers['linepts'].append(linepts)
        
    return tracers

#

def within_plant(points, foliage_points, skipz=False):
    
    keep = np.ones(points.shape[0], dtype=bool)
    if skipz: n = 2
    else: n = 3
    for i in range(n):
        keep &= np.logical_and(points[:,i] >= foliage_points[:,i].min(),
                               points[:,i] <= foliage_points[:,i].max())

    return keep

def get_beams_traces(foliage_points, files, res=0.1):
    """
    Get beam traces points within plant region.
    """

    beam_points_wpr = []
    # read the numpy files
    for file in files:

        df = np.loadtxt(file)
        if len(df) == 0: continue
        filename = file.split('/')[-1]
#         print(filename)
        points = df.T[5:8].T # x,y, and z coordinates
        beam_angles = df.T[1:3].T # Pitch and Yaw
        distance = df.T[3:4].T # distance from UAV to intercepted point
        
        #continue if points within plant region
        inplant = within_plant(points, foliage_points)
        if inplant.sum() == 0: 
            print(filename, 'skipped...')
            continue

        tracers = trace_beam(points[inplant], beam_angles[inplant], distance[inplant], res=res)
        beam_points = np.vstack(tracers['linepts'])
        beam_points = beam_points.reshape(beam_points.shape[0], beam_points.shape[2])
    #     print(beam_points.shape)

        # filter within plant region
#         keep = np.ones(beam_points.shape[0], dtype=bool)
#         for i in range(3):
#             keep &= np.logical_and(beam_points[:,i] >= foliage_points[:,i].min(),
#                                    beam_points[:,i] <= foliage_points[:,i].max())
        keep = within_plant(beam_points, foliage_points)

        print('%s \t beam traces points: \t %i --> %i' %(filename, len(keep), np.sum(keep)))
        if np.sum(keep) > 0:
            beam_points_wpr.append(beam_points[keep])

    return beam_points_wpr

#

def points2pcd(points):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=np.asarray(pcd.points).shape))
    
    return pcd

def get_normals(points, kd3_sr, max_nn, show=False, downsample=False):
    
    pcd = points2pcd(points)
    
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=kd3_sr,max_nn=max_nn))
    
    if downsample:
        
        downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.01)

        if show:
            o3d.visualization.draw_geometries([downpcd])
        
        return np.asarray(downpcd.normals), np.asarray(downpcd.points)

    else:
        
        return np.asarray(pcd.normals)


#

def vecttoangle(v1, v2):
    
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    
    return np.rad2deg(angle)

def get_leaf_angle(points, normals, voxel_size, show=False, norm_avg=True):
    
    t1 = process_time()

    if norm_avg:
    
        pcd = points2pcd(points)
        
        # Voxelizing the PCD
        # print('voxelization')
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
            
        # get voxel ID for each point
        voxel = []
        queries = points

        for vox in queries:

            i,j,k = voxel_grid.get_voxel(vox)
            voxel.append('%s_%s_%s' %(i,j,k))

        # print('N voxels: %i' %(len(set(voxel))))
        # print('Voxel size: \t %.4f' %(voxel_size))
        
        # 
        angs = {'voxID':[], 'avgVect':[], 'avgAngle':[], 'N':[]}

        #
        vox, idx, idxinv, counts = np.unique(np.array(voxel), return_index=True, return_inverse=True, return_counts=True)

        for num, vx in enumerate(vox):

            keep = np.where(idxinv == num)

            angs['voxID'].append(vx)
            angs['N'].append(counts[num])

            # average all normals within voxel
            avgVect = normals[keep].sum(axis=0)
            angs['avgVect'].append(avgVect)

            # get angle with respect to z axis from average normals
            # ang = vecttoangle([0, 0, 1], np.abs(avgVect))
            ang = vecttoangle([0, 0, 1], -avgVect)
            angs['avgAngle'].append(np.round(ang, 1))

        if show:
            o3d.visualization.draw_geometries([voxel_grid])

    else:
        print('Option: not norms averaged...')
        angs = {'avgAngle':[]}

        for normal in normals:

            ang = vecttoangle([0, 0, 1], -normal)
            angs['avgAngle'].append(np.round(ang, 2))

    t2 = process_time()
    # print('Total time:', t2-t1)
        
    return angs
    
#
def get_voxels(voxel_grid):
    
    voxels = []
    for i in voxel_grid.get_voxels():
#         voxels.append('_'.join(str(j) for j in i.grid_index.tolist()))
        voxels.append(i.grid_index.tolist())
    
    return np.array(voxels)

def idx_bounds(voxel_idx, show=False):
    
    # get i,j,k max and min
    vdict = {}
    vdict['ijk_max'] = [voxel_idx.T[i].max() for i in [0,1,2]]
    vdict['ijk_min'] = [voxel_idx.T[i].min() for i in [0,1,2]]
    
    if show:
        print('max -->', vdict['ijk_max'])
        print('min -->', vdict['ijk_min'])
        
    return vdict

#
def within_bounds(voxel_idx, voxel_idx_bounds):
    
    vdict = idx_bounds(voxel_idx_bounds, False)
    
    keep = np.ones(voxel_idx.shape[0], dtype=bool)
    for i in range(3):
        keep &= np.logical_and(vdict['ijk_min'][i] <= voxel_idx[:,i], vdict['ijk_max'][i] >= voxel_idx[:,i])
        
    # Check 3D matrix i,j,k size matches length of solid voxel grid index array: i*j*k == voxs_idx.shape[0]
    vdict = idx_bounds(voxel_idx[keep], False)
    assert np.product(np.array(vdict['ijk_max'])+1) == voxel_idx_bounds.shape[0]
        
    return voxel_idx[keep]

#
def get_attributes(m3shape, m3p, m3b, show=False):
    
    # 3D matrix full with attribute 3
    m3att = np.full(m3shape, 3, dtype=int)
    
    # fill with attribute 2
    m3att[m3b] = 2
    
    # fill with attribute 1
    m3att[m3p] = 1
    
    assert (m3att == 3).sum() == (~(m3b & (~m3p) | (m3p))).sum()
    assert (m3att == 2).sum() == (m3b & (~m3p)).sum()
    assert (m3att == 1).sum() == m3p.sum()
    
    if show:
        for i in range(1,4):
            print('Number of voxels with attribute %i: \t %i' %(i, (m3att == i).sum()))

    return m3att

#
def stheta(theta, thetaLq):
    
    x = lambda theta, thetaLq : 1/np.cos(np.arctan(np.tan(theta))*np.arctan(np.tan(thetaLq)))
    
    if theta <= np.pi/2 - thetaLq:
        s = np.cos(theta)*np.cos(thetaLq)
        
    elif theta > np.pi/2 - thetaLq:
        s = np.cos(theta)*np.cos(thetaLq)*(1 + 2*(np.tan(x(theta, thetaLq)) - x(theta, thetaLq))/np.pi)
    else:
        raise ValueError('theta=%.2f and/or thetaL=%.2f fails condition' %(theta, thetaLq))
        
    return s
    
def Gtheta(theta, thetaL, tq, norm=True):
    
    """
    Compute the G(\theta).
    
    theta:  float::  beam-angle in degrees
    thetaL: float-array:: leaf-inclination-angle distribution in degrees
    tq: int:: total number of leaf-inclination-angle classes
    """
    
    bins = np.linspace(0, 90, tq+1)
    gtot = len(thetaL)
    Gtheta_i = []

    for q in range(len(bins)-1):
        
        keep = np.logical_and(thetaL >= bins[q], thetaL < bins[q+1])
        if norm: gqi = np.sum(keep)/gtot # normalize the inclination-angle distribution?
        else: gqi = np.sum(keep)
        thetaLq = np.median(np.array(thetaL)[keep])
        if np.isnan(thetaLq):
            thetaLq = bins[q] + (bins[q+1] - bins[q])/2.
#         print(q, thetaLq)
        sthetaq = stheta(np.radians(np.array(theta)), np.radians(np.array(thetaLq)))
        Gtheta_i.append(gqi*sthetaq)

#         print(gqi*sthetaq)
#     print('Total: ',np.array(Gtheta_i).sum())

    return np.array(Gtheta_i).sum()
    
#
def get_lad_perk(kcoord, m3att, alpha, voxel_size):
    
    ki, kf = kcoord
    
    if kf > m3att.shape[2]:
        raise ValueError('k values cannot be greater than available. Maximum K value is: %i' %(m3att.shape[2]))
    
    m3 = m3att[:,:,ki:kf]
    DeltaH = (kf-ki) * voxel_size
    lai = []
    
    for i in range(kf-ki):
        
        nI = (m3[:,:,i] == 1).sum()
        nP = (m3[:,:,i] == 2).sum()
        lai.append(nI/(nI+nP))
#         print(i, nI, nP, nI/(nI+nP))
#         print(i, nI/(nI+nP), DeltaH)
        
    LAD = alpha * (1/DeltaH) * np.array(lai).sum()
#     print(alpha, 1/DeltaH, np.array(lai).sum(), LAD)
#     print(ki*voxel_size+DeltaH/2., LAD)

    return ki*voxel_size+DeltaH/2., LAD
    
#
def get_LADS(m3att, alpha, voxel_size, kbins):
    
    kmax = m3att.shape[2]
    ar = np.arange(0, kmax, kbins)
    kcoords = []
    lads = []

    for i in range(len(ar)-1):
        kcoords.append([ar[i],ar[1:][i]])

    kcoords.append([ar[-1], kmax])
    
#     print(kcoords)
    for i in kcoords:
        h, lad = get_lad_perk(i, m3att, alpha, voxel_size)
        lads.append([h, lad])
        
    return np.array(lads)


#
# file = "../data/mock_vel2/toy_trees.ply"
def true_lad(file, DeltaH, leafang):
    
    mesh = o3d.io.read_triangle_mesh(file)
    bxi, byi, bzi = mesh.get_min_bound()
    bxf, byf, bzf = mesh.get_max_bound()
    lx, ly = np.abs(bxf - bxi), np.abs(byf - byi)
     
    #divide heigth in bins
    ar = np.arange(bzi, bzf, DeltaH)
    kcoords = []
    for i in range(len(ar)-1):
        kcoords.append([ar[i],ar[1:][i]])
    kcoords.append([ar[-1], bzf])
    
    #compute true lad per bin
    lads = []
    # crop mesh for each bin
    for i in kcoords:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(bxi, byi, i[0]), max_bound=(bxf, byf, i[1]))
        _mesh = mesh.crop(bbox)
        # get surface of croped mesh
        A = _mesh.get_surface_area()
        # correct by leaf inclination
        Areal = A*np.cos(np.radians(leafang))
#         print('%.2f < z < %.2f: \t A=%.2f \t Areal=%.2f' %(i[0], i[1], A, Areal))
        
        #get volume per bin
        lz = np.abs(i[1] - i[0])
        volume = lx * ly * lz
        
        #save lad
        lads.append([i[0]+lz/2., Areal/volume])
        
    return np.array(lads)

def true_angles(file):

    mesh = o3d.io.read_triangle_mesh(file)
    mesh.compute_triangle_normals()
    tnorm = np.asarray(mesh.triangle_normals)

    true_angs = []

    for vect in tnorm:

        # angs = np.round(vecttoangle([0, 0, 1], np.abs(vect)), 1)
        angs = np.round(vecttoangle([0, 0, 1], vect), 1)
        true_angs.append(angs)

    return true_angs

def correct_angs(angs):

    thetaL = [i if (i < 90) else (180 - i) for i in angs]

    return thetaL

def mesh_leaf_area(meshfile):

    # get leaf area
    mesh = o3d.io.read_triangle_mesh(meshfile)
    cidx, nt, area = mesh.cluster_connected_triangles()

    # Hexagonal leaves from blensor have 4 trinagles in mesh
    keep = (np.array(nt) == 4)
    if keep.sum() != 0:
        la = np.array(area)[keep][0]
    else:
        raise ValueError('Mesh does not find clusters leaves with 4 triangles.')

    return np.round(la, 5)

def get_weigths(points, voxel_size=0.5):

    pcd = points2pcd(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxel = []

    # for each point, get the voxel index
    for point in points:

        i,j,k = voxel_grid.get_voxel(point)
        voxel.append('%s_%s_%s' %(i,j,k))

    # get voxel list, indexes, and counts of points per voxel
    vox, idx, idxinv, counts = np.unique(np.array(voxel), return_index=True, return_inverse=True, return_counts=True)
    # Voxel volume
    volume = voxel_size**3
    # Point cloud volume density per voxel
    density = counts/volume
    # Mean volume density
    mean_density = np.mean(density)
    # Weigth value per voxel
    w = density / mean_density
    # Weigth value per point
    ws = w[idxinv]

    return ws

def test_leaf_angle(mockname, voxel_size_la, voxel_size_w, kd3_sr, max_nn, debug=False, 
savefig=None, text=None, norm_avg=True, downsample=False, weigths=True):

    mockdir = os.path.join(_data, mockname)
    segtrees_dir_name = 'toy_trees'
    segtrees_dir = os.path.join(mockdir, segtrees_dir_name)

    # Results directory
    resdir_name = '%s_%s' %('results', mockname)
    resdir = os.path.join(mockdir, resdir_name)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    # path of files
    segtrees_files = glob.glob(os.path.join(segtrees_dir, 'tree_*.npy'))
    # rawdata_files = glob.glob(os.path.join(mockdir, 'toy0*.numpy'))

    # Mesh file
    meshfile = os.path.join(mockdir, 'mesh.ply')
    if os.path.isfile(meshfile):
        ta = true_angles(meshfile)
    else:
        raise ValueError('No mesh.ply file in %s' %(mockdir))

    if debug:
        # segtrees_files = segtrees_files[3:4]
        segtrees_files = segtrees_files[0:1]

    chis2 = []

    for file in segtrees_files:


        # Tree name
        treename = file.split('/')[-1].split('.')[0]
        # load segmented tree (foliage only) data
        tree = np.load(file)
        if debug:
            tree = tree[::10]
        # Extract x,y, and z coordinates of foliage point cloud (fpc)
        points = tree.T[5:8].T 

        # Reconstruct the discrete trace beam (tpc)
        # tracers = get_beams_traces(points, rawdata_files, res=rt_res)
        # tracers = np.vstack(tracers)
        # print('Tracers dimensions: \t', tracers.shape)

        # Compute the normals to fpc (nfpc)
        if downsample:
            normals, points = get_normals(points, kd3_sr, max_nn, show=False, downsample=downsample)
        else:
            normals = get_normals(points, kd3_sr, max_nn, show=False, downsample=downsample)

        # Get the Leaf Inclination Angle
        angs = get_leaf_angle(points, normals, voxel_size_la, show=False, norm_avg=norm_avg)
        thetaL = angs['avgAngle']

        # -----DEPRECATED-----
        # correct \theta_{L}
        # if correct_theta:

        #     thetaL = correct_angs(angs['avgAngle'])
        #     true_angles = correct_angs(true_angles)
            
        # else:
        #     thetaL = angs['avgAngle']
        thetaL = correct_angs(angs['avgAngle'])
        ta = correct_angs(ta)

        if savefig is None:
            savefig = os.path.join(resdir, 'leaf_angle_dist_%s.png' %(treename))

        if weigths:
            ws = get_weigths(points, voxel_size=voxel_size_w)
        else:
            ws = None

        h, htruth = figures.angs_dist(thetaL, ta, savefig=savefig, text=text, ws=ws)
        # find the chisquare
        # h = [np.nan if x == 0 else x for x in h]
        # htruth = [np.nan if x == 0 else x for x in htruth]
        if float(0) in htruth:
            chi2, p = chisquare(h+1, htruth+1)
        else:
            chi2, p = chisquare(h, htruth)
        # print(h+1, htruth+1)
        # print(chi2, p)
        chis2.append([treename, chi2, p])

    if debug:
        return chis2[0]
    else:
        return chis2

def bestfit_pars_la(mockname, norm_avg=True, downsample=False, weigths=True):

    mockdir = os.path.join(_data, mockname)
    resdir_name = '%s_%s' %('results', mockname)
    outfile = os.path.join(mockdir, resdir_name, 'bestfit.npy')

    # voxel_size_la = 0.01
    # kd3_sr=0.05
    # max_nn=10
    voxel_size_la = 0.1
    voxel_size_w = 0.01
    kd3_sr = 1
    max_nn = 5

    pars = {}
    if norm_avg:
        pars['voxel_size_la'] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    pars['voxel_size_w'] = [0.0001, 0.001, 0.01, 0.1, 1]
    pars['kd3_sr'] = [0.001, 0.01, 0.1, 1.0]
    pars['max_nn'] = [3, 5, 10, 20, 50, 100]
    # pars['voxel_size_la'] = [0.01, 0.05, 0.1]
    # pars['kd3_sr'] = [0.1, 0.5,  1.0]
    # pars['max_nn'] = [3, 10, 50]

    # Mesh file
    meshfile = os.path.join(mockdir, 'mesh.ply')

    if os.path.isfile(meshfile):
        la = mesh_leaf_area(meshfile)
    else:
        raise ValueError('No mesh.ply file in %s' %(mockdir))

    res = {}
    res['leafsize'] = la
    if norm_avg:
        res['voxel_size_la_fixed'] = voxel_size_la
    res['voxel_size_w_fixed'] = voxel_size_w
    res['kd3_sr_fixed'] = kd3_sr
    res['max_nn_fixed'] = max_nn

    for key, val in pars.items():

        res[key] = []

        for par in val:

            if key == 'voxel_size_la':
                chis2 = test_leaf_angle(mockname, par, voxel_size_w, kd3_sr, max_nn, debug=True, norm_avg=norm_avg, downsample=downsample, weigths=weigths)
            elif key == 'voxel_size_w':
                chis2 = test_leaf_angle(mockname, voxel_size_la, par, kd3_sr, max_nn, debug=True, norm_avg=norm_avg, downsample=downsample, weigths=weigths)
            elif key == 'kd3_sr':
                chis2 = test_leaf_angle(mockname,  voxel_size_la, voxel_size_w, par, max_nn, debug=True, norm_avg=norm_avg, downsample=downsample, weigths=weigths)
            elif key == 'max_nn':
                chis2 = test_leaf_angle(mockname,  voxel_size_la, voxel_size_w, kd3_sr, par, debug=True, norm_avg=norm_avg, downsample=downsample, weigths=weigths)
            else:
                raise ValueError('%s is not a valid parameter' %(key))

            res[key].append([chis2[0], par, chis2[1]])
            print(key, par, 'DONE...')

        df = pd.DataFrame(res[key], columns=['tree', 'value', 'chi2'])
        keep = (df['chi2'] == df['chi2'].min())
        bestfit_par = df.loc[keep, 'value'].values[0]

        res[key+'_'+'bestfit'] = bestfit_par
        print(key, 'BESTFIT:\t', bestfit_par)

    # save dict
    np.save(outfile, res)

    return res  

def best_fit_pars_plot(res, mockname, savefig=None, norm_avg=True, downsample=False, weigths=True):

    if norm_avg:
        pars = ['voxel_size_la', 'voxel_size_w', 'kd3_sr', 'max_nn']
    else:
        pars = ['voxel_size_w', 'kd3_sr', 'max_nn']

    fig = plt.figure(figsize=(15, 4*len(pars)))

    for num, par in enumerate(pars):

        plt.subplot(len(pars),1,num+1)

        x = np.array(res[par])[:,1].astype('float')
        y = np.array(res[par])[:,2].astype('float')

        if par in ['voxel_size_la', 'voxel_size_w', 'kd3_sr']:
            plt.semilogx(x, y, lw=3, marker='*')
        else:
            plt.plot(x, y, lw=3, marker='*')

        plt.axvline(res[par+'_'+'bestfit'], lw=2, ls='--', color='g')
        
        plt.xlabel(par)
        plt.ylabel(r'$\chi^2$')

    if savefig is None:

        mockdir = os.path.join(_data, mockname)
        resdir_name = '%s_%s' %('results', mockname)
        savefig = os.path.join(mockdir, resdir_name, 'bestfits_pars.png')

    plt.savefig(savefig, dpi=200, bbox_inches='tight')
