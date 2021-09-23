
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


def beam_direction(yaw, pitch, scan_inc):
    '''
    Get laser beam direction from YAW and PITCH
    '''
    
    x = np.cos(yaw)*np.cos(pitch+np.radians(scan_inc - 90))
    y = np.sin(yaw)*np.cos(pitch+np.radians(scan_inc - 90))
    z = np.sin(pitch+np.radians(scan_inc - 90))
    
    points = np.vstack((np.array(x), np.array(y), np.array(z))).transpose()
    
    return points

def trace_beam(points, beam_angles, distance, scan_inc, res=0.1):

    tracers = {'p':[], 'pb':[], 'linepts':[]}

#     beam_angles = np.vstack((np.array(df['yaw']), np.array(df['pitch']))).transpose()
#     dist = np.array(df['distance'])

    # unitary vector with direction of beam from camera point of view
    uv = beam_direction(beam_angles.T[0], beam_angles.T[1], scan_inc) 
    # print(uv[:5])
    for i in range(uv.shape[0]):
        
        # proyect unitary vector to actual distance of beam and invert it to point cloud data view
        N = int(distance[i]/res)
        pb = 1 * uv[i] * np.linspace(0.01, distance[i], N)[:, np.newaxis] # this traces the beam with N points

        # Moving proyected unitary vector (i.e. the beam) to its real position
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

def get_beams_traces(foliage_points, files, scan_inc, res=0.1):
    """
    Get beam traces points within plant region.
    """

    beam_points_wpr = []
    # read the numpy files
    for file in files:

        df = np.load(file)
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

        tracers = trace_beam(points[inplant], beam_angles[inplant], distance[inplant], scan_inc, res=res)
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

def get_all_traces(mockname, res=0.1, debug=False, downsample_debug=1):
    """
    Get all beam traces points for each segmented tree within plant region.
    """

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
    rawdata_files = glob.glob(os.path.join(mockdir, 's*.npy'))

    tracers_dict = {i.split('/')[-1].split('.')[0]:[] for i in segtrees_files}

    # read the numpy files
    for file in rawdata_files:

        df = np.load(file)

        if len(df) == 0: 
            continue
        else:
            print(file)
            # print(df.T[1:3].T2[:3])

        if debug:
            df = df[::downsample_debug]

        filename = file.split('/')[-1]
        # x, y, z = df.T[5], df.T[6], df.T[7]
        points = df.T[5:8].T

        # get sensor coordinates
        try:
            spos = os.path.join(mockdir, 'scanner_pos.txt')
        except Exception as e:
            raise ValueError(e)

        scan = laod_scan_pos(spos)
        id = [i.decode("utf-8") for i in scan['scan']]
        keep = np.array(id) == filename[:2]
        _, sx, sy, sz = scan[keep][0]

        # sensor coordinates
        p2 = [sx, sy, sz]

        # for each LiDAR point, draw a point-like line trhough the sensor
        tracers = []
        for p1 in points:

            pp = line2points_vect(p1, p2, res=res)
            tracers.append(pp)

        beam_points = np.vstack(tracers)
        # beam_points = beam_points.reshape(beam_points.shape[0], beam_points.shape[2])

    # ''' 
        for treefile in segtrees_files:

            # Tree name
            treename = treefile.split('/')[-1].split('.')[0]
            # load segmented tree (foliage only) data
            tree = np.load(file)
            # Extract x,y, and z coordinates of foliage point cloud (fpc)
            foliage_points = tree.T[5:8].T

            #continue if points within plant region
            inplant = within_plant(beam_points, foliage_points)

            if inplant.sum() == 0: 
                print(treename, 'skipped...')
                continue
            else:
                print('%s \t beam traces points in tree %s: \t %i --> %i' %(filename, treename, points.shape[0], np.sum(inplant)))
                tracers_dict[treename].append(beam_points[inplant])

    for key, val in tracers_dict.items():

        outdir = os.path.join(mockdir, segtrees_dir_name, 'tracers_%s_%s.npy' %(key, str(res)))
        new_val = np.array(np.vstack(val))
        np.save(outdir, new_val)
    
    # '''

    return tracers

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
    
    # x = lambda theta, thetaLq : 1/np.cos(np.arctan(np.tan(theta))*np.arctan(np.tan(thetaLq)))
    x = lambda theta, thetaLq : np.arccos((1/np.tan(theta))*(1/np.tan(thetaLq)))
    
    # s = np.cos(theta)*np.cos(thetaLq)

    if theta <= np.pi/2 - thetaLq:
        s = np.cos(theta)*np.cos(thetaLq)
        
    elif theta > np.pi/2 - thetaLq:
        s = np.cos(theta)*np.cos(thetaLq)*(1 + 2*(np.tan(x(theta, thetaLq)) - x(theta, thetaLq))/np.pi)
    else:
        raise ValueError('theta=%.2f and/or thetaL=%.2f fails condition' %(theta, thetaLq))
        
    return s
    
# def Gtheta(theta, thetaL, tq, norm=True):
    
#     """
#     Compute the G(\theta).
    
#     theta:  float::  beam-angle in degrees
#     thetaL: float-array:: leaf-inclination-angle distribution in degrees
#     tq: int:: total number of leaf-inclination-angle classes
#     """
    
#     bins = np.linspace(0, 90, tq+1)
#     gtot = len(thetaL)
#     Gtheta_i = []

#     for q in range(len(bins)-1):
        
#         keep = np.logical_and(thetaL >= bins[q], thetaL < bins[q+1])
#         if norm: gqi = np.sum(keep)/gtot # normalize the inclination-angle distribution?
#         else: gqi = np.sum(keep)
#         thetaLq = np.median(np.array(thetaL)[keep])
#         if np.isnan(thetaLq):
#             thetaLq = bins[q] + (bins[q+1] - bins[q])/2.
# #         print(q, thetaLq)
#         sthetaq = stheta(np.radians(np.array(theta)), np.radians(np.array(thetaLq)))
#         Gtheta_i.append(gqi*sthetaq)

# #         print(gqi*sthetaq)
# #     print('Total: ',np.array(Gtheta_i).sum())

#     return np.array(Gtheta_i).sum()

def Gtheta(theta, thetaLq, gq):
    
    """
    Compute the G(\theta).
    
    theta:  float::  beam-angle in degrees
    thetaL: float-array:: leaf-inclination-angle distribution in degrees
    """

    Gtheta_i = []

    for q, i in zip(thetaLq, gq):
        
        sthetaq = stheta(np.radians(np.array(theta)), np.radians(np.array(q)))
        Gtheta_i.append(i*sthetaq)
        # print(i, sthetaq)

    return np.array(Gtheta_i).sum()
    
#
def get_lad_perk(kcoord, m3att, alphas, voxel_size, alpha2):
    
    ki, kf = kcoord
    # print(kf-ki)
    
    if kf > m3att.shape[2]:
        raise ValueError('k values cannot be greater than available. Maximum K value is: %i' %(m3att.shape[2]))
    
    m3 = m3att[:,:,ki:kf]
    DeltaH = (kf-ki) * voxel_size
    lai = []
    
    for i in range(kf-ki):
        
        nI = (m3[:,:,i] == 1).sum()
        nP = (m3[:,:,i] == 2).sum()
        _lai = nI/(nI+nP)
        alpha = alphas[i]
        # print(1/DeltaH, alpha, _lai)
        # lai.append(_lai)
        lai.append(alpha * _lai)
#         print(i, nI, nP, nI/(nI+nP))
#         print(i, nI/(nI+nP), DeltaH)
        
    # LAD = alpha2 * np.mean(alphas) * (1/DeltaH) * np.array(lai).sum()
    LAD = alpha2 * (1/DeltaH) * np.array(lai).sum()

#     print(alpha, 1/DeltaH, np.array(lai).sum(), LAD)
    # print('k, mean alphas: ', kcoord, np.mean(alphas))
    # print((ki+(kf-ki-1)/2)*voxel_size, LAD)

    return (ki+(kf-ki-1)/2)*voxel_size, LAD
    
#
def get_LADS(m3att, voxel_size, kbins, alphas_k, alpha2):
    
    kmax = m3att.shape[2]
    ar = np.arange(0, kmax, kbins)
    kcoords = []
    lads = []

    for i in range(len(ar)-1):
        kcoords.append([ar[i],ar[1:][i]])

    kcoords.append([ar[-1], kmax])
    
#     print(kcoords)
    for i in kcoords:
        ki, kf = i
        alphas = alphas_k[ki:kf]
        # print(i, alphas)
        h, lad = get_lad_perk(i, m3att, alphas, voxel_size, alpha2)
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

    return np.round(la, 6)
    # return la

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

def get_voxk(points, voxel_size=0.5, mesh=False):

    pcd = points2pcd(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxel = []

    # for each point, get the voxel index
    for point in points:

        i,j,k = voxel_grid.get_voxel(point)
        voxel.append(k)

    return voxel

def get_voxk_mesh(meshfile, voxel_size=0.5):

    mesh = o3d.io.read_triangle_mesh(meshfile)
    vert = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)

    # Get mesh by height from vertices points
    voxk = get_voxk(vert, voxel_size=voxel_size)

    voxel = []
    # For each traingle, get its corresponfing voxel k (height)
    # one for each of the three vertices that form the triangle
    # and keep the most frequent K
    for i in tri:
        
        a = [voxk[i[0]], voxk[i[1]], voxk[i[2]]]
        counts = np.bincount(np.array(a))
        voxel.append(np.argmax(counts))

    return voxel

def test_leaf_angle(mockname, voxel_size_la, voxel_size_w, kd3_sr, max_nn, debug=False, 
savefig=None, text=None, norm_avg=True, downsample=False, weigths=True, voxel_size_h=1, downsample_debug=10):

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
        voxk_mesh = get_voxk_mesh(meshfile, voxel_size=voxel_size_h)
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
            tree = tree[::downsample_debug]
        else:
            downsample_debug = None
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
            _savefig = os.path.join(resdir, 'leaf_angle_dist_%s.png' %(treename))
            _savefig_k = os.path.join(resdir, 'leaf_angle_dist_height_%s.png' %(treename))
            _savefig_G = os.path.join(resdir, 'G_alpha_theta_%s.png' %(treename))
        else:
            _savefig = os.path.join(resdir, 'leaf_angle_dist_%s_%s.png' %(treename, savefig))
            _savefig_k = os.path.join(resdir, 'leaf_angle_dist_height_%s_%s.png' %(treename, savefig))
            _savefig_G = os.path.join(resdir, 'G_alpha_theta_%s_%s.png' %(treename, savefig))

        if weigths:
            ws = get_weigths(points, voxel_size=voxel_size_w)
        else:
            ws = None

        h, thetaLq, htruth = figures.angs_dist(thetaL, ta, savefig=_savefig, text=text, ws=ws, downsample_debug=downsample_debug)
        voxk = get_voxk(points, voxel_size=voxel_size_h)
        figures.angs_dist_k(voxk, voxk_mesh, thetaL, ta, ws=ws, savefig=_savefig_k)

        # # get G(theta)
        # thetaLq = (thetaLq[:-1]+thetaLq[1:])/2
        # # print(thetaLq, h, htruth)
        # Gthetas = np.array([[i, Gtheta(i, thetaLq, h)] for i in np.arange(0, 95, 2)])
        # alpha = np.cos(np.radians(Gthetas[:,0]))/Gthetas[:,1]
        # G_alpha_theta = np.vstack((Gthetas[:,0], Gthetas[:,1], alpha))
        # figures.G_alpha_plot(G_alpha_theta, savefig=_savefig_G)
        # outdir_g = os.path.join(mockdir, segtrees_dir_name, 'G_alpha_theta_%s_%s.npy' %(treename, savefig))
        # np.save(outdir_g, G_alpha_theta)

        # Save angles and weights?
        # not entirely sure because file will be particular
        # to a downsample_debug
        outdir_angs = os.path.join(mockdir, segtrees_dir_name, 'angles_%s_%s.npy' %(treename, str(downsample_debug)))
        np.save(outdir_angs, thetaL)
        if weigths:
            outdir_ws = os.path.join(mockdir, segtrees_dir_name, 'weights_%s_%s.npy' %(treename, str(downsample_debug)))
            np.save(outdir_ws, ws)

        if float(0) in htruth:
            chi2, p = chisquare(h+1, htruth+1)
        else:
            chi2, p = chisquare(h, htruth)

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


def get_lad(mockname, res=0.05, debug=True, downsample_debug=None, voxel_size=0.1):

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
    rawdata_files = glob.glob(os.path.join(mockdir, 'toy*.npy'))

    # Mesh file
    meshfile = os.path.join(mockdir, 'mesh.ply')
    if os.path.isfile(meshfile):
        print(meshfile)
        # ta = lad.true_angles(meshfile)
        # voxk_mesh = lad.get_voxk_mesh(meshfile, voxel_size=voxel_size_h)
    else:
        raise ValueError('No mesh.ply file in %s' %(mockdir))

    if debug:
        # segtrees_files = segtrees_files[3:4]
        segtrees_files = segtrees_files[0:1]

    # chis2 = []

    for file in segtrees_files:

        t1 = process_time()

        # Tree name
        treename = file.split('/')[-1].split('.')[0]

        # Check if tracers are available for this tree
        tracer_file = os.path.join(mockdir, segtrees_dir_name, 'tracers_%s_%s.npy' %(treename, str(res)))
        if os.path.isfile(tracer_file):
            tracers = np.load(tracer_file)
        else:
            raise ValueError('No such file: %s' %(tracer_file))

        # Check if angles and weights are available
        outdir_angs = os.path.join(mockdir, segtrees_dir_name, 'angles_%s_%s.npy' %(treename, str(1)))
        outdir_ws = os.path.join(mockdir, segtrees_dir_name, 'weights_%s_%s.npy' %(treename, str(1)))

        # load segmented tree (foliage only) data
        tree = np.load(file)
        if debug:
            tree = tree[::downsample_debug]
        else:
            downsample_debug = None
            
        # Extract x,y, and z coordinates of foliage point cloud (fpc)
        points = tree.T[5:8].T
        Spoints = tree.T[16:19].T
        # beam_angles = tree.T[1:3].T # Pitch and Yaw

        t2 = process_time()
        print('Stage 1:', t2-t1)

        t1 = process_time()

        # find atributes

        # get points voxel bounding box
        pcd = points2pcd(points)
        voxp = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        # voxp.get_axis_aligned_bounding_box()

        # Voxelize the beam points with same bounding box dimensions as the points voxel grid
        pcd = points2pcd(tracers)
        voxb = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=voxel_size, 
                                                                            min_bound=voxp.get_min_bound().reshape(3,1), 
                                                                            max_bound=voxp.get_max_bound().reshape(3,1))

        # Create voxel of plant region
        width, height, depth = voxp.get_max_bound() - voxp.get_min_bound()
        print(width, height, depth)
        voxs = o3d.geometry.VoxelGrid.create_dense(origin=voxp.origin, color=np.array([0,0,1]), voxel_size=voxel_size, width=width, height=height, depth=depth)

        t2 = process_time()
        print('Stage 2:', t2-t1)

        t1 = process_time()

        # Get solid voxel grid indexes
        voxs_idx = get_voxels(voxs)

        # get i,j,k max and min
        vdict = idx_bounds(voxs_idx, True)
        ijk_bounds = np.array(list(vdict.values())).reshape(1,6)[0]

        # check that all index (i,j and k) are positive
        if np.any(ijk_bounds < 0):
            raise ValueError('Solid voxel grid found negative (i,j,k) values.')

        # create 3D boolean matrix of i,j,k size
        m3s = np.zeros(np.array(vdict['ijk_max'])+1, dtype=bool)
        print(m3s.shape)

        # Check 3D matrix i,j,k size matches length of solid voxel grid index array: i*j*k == voxs_idx.shape[0]
        assert np.product(np.array(vdict['ijk_max'])+1) == voxs_idx.shape[0]

        t2 = process_time()
        print('Stage 3:', t2-t1)

        t1 = process_time()

        # get voxel grid indexes for points and beams voxels
        voxp_idx = get_voxels(voxp)
        voxb_idx = get_voxels(voxb)

        # get rid of voxels outside the plant region
        voxp_idx = within_bounds(voxp_idx, voxs_idx)
        voxb_idx = within_bounds(voxb_idx, voxs_idx)

        # fill 3D matrix with True if voxel exist
        m3p = m3s.copy()
        for (i,j,k) in voxp_idx:
            
            m3p[i][j][k] = True
            
        print('Number of voxels ocupied by points cloud: \t %i' %(m3p.sum()))

        m3b = m3s.copy()
        for (i,j,k) in voxb_idx:
                
            m3b[i][j][k] = True
            
        print('Number of voxels ocupied by beam points cloud: \t %i' %(m3b.sum()))
        print('Total number of voxels in plant regions: \t %i' %((~m3s).sum()))

        m3att = get_attributes(m3s.shape, m3p, m3b, True)

        t2 = process_time()
        print('Stage 4:', t2-t1)

        t1 = process_time()

        # get LAD
        voxk = get_voxk(points, voxel_size)
        lia = np.load(outdir_angs)
        lia = lia[::downsample_debug]
        ws = np.load(outdir_ws)
        ws = ws[::downsample_debug]

        # get beam inclination angle (BIA)
        bia = []
        for i, j in zip(points, Spoints):
            # beam vector
            v = np.array(j) - np.array(i)
            # beam unitary vector
            uv = v / np.linalg.norm(v)
            bia.append(vecttoangle([0, 0, 1], uv))


        # alphas_k = alpha_k(bia, voxk, lia, ws, resdir)

        # lads = get_LADS(m3att, alpha, voxel_size, kbins)
        # kmax = m3att.shape[2]
        # lads_mesh = get_LADS_mesh(meshfile, voxel_size, kbins, kmax)
        t2 = process_time()
        print('Stage 5:', t2-t1)

    return m3att, meshfile, bia, voxk, lia, ws, resdir

def alpha_k(bia, voxk, lia, ws, resdir, show=False):

    colors = plt.cm.jet(np.linspace(0,1,len(set(voxk))))
    # uv = beam_direction(beam_angles.T[0], beam_angles.T[1], scan_inc)
    
    bins = np.linspace(0, 90, int(90/1)) # Don't change this!!!
    weights = 1/ws
    h, x = np.histogram(lia, bins=bins, weights=weights, density=True)
    thetaLq = (x[:-1]+x[1:])/2
    alpha = [np.cos(np.radians(i))/Gtheta(i, thetaLq, h) for i in range(90)]
    alpha_f = lambda theta: np.cos(np.radians(theta))/Gtheta(theta, thetaLq, h)

    bia_ = bia
    bia = np.array(correct_angs(np.array(bia_)))
    bamin, bamax = np.percentile(bia, (0.3,99.7))
    ba = np.linspace(bamin, bamax, len(set(voxk)))

    if show:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(1,1,1)
        plt.hist(np.array(bia_), 40, histtype='step', label='pre correction')
        plt.hist(bia, 40, histtype='step', label='after correction')
        plt.legend()
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$Frecuency$')

        savefig = os.path.join(resdir, 'bia.png')
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

    if show:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(1,1,1)

        for k in list(set(voxk)):
            keep = voxk == k
            angi = bia[keep]
            median = np.median(angi)

            plt.subplot(1,1,1)
            if k == 0 or k == list(set(voxk))[-1]:
                label = 'k=%i' %(k)
            else:
                label = None
            plt.hist(np.array(angi), 30, histtype='step', color=colors[k], label=label)
            # plt.axvline(median, ls='--')
            plt.legend()

        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$Frecuency$')
        savefig = os.path.join(resdir, 'bia_per_k.png')
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

    # heigh density
    # bins_k = list(set(voxk))
    # bins_ = np.linspace(0, len(bins_k), len(bins_k)+1)
    # hd, kb = np.histogram(voxk, bins=bins_, weights=weights, density=True)
    # print(kb, hd)

    res = []
    if show:
        fig = plt.figure(figsize=(14, 6))

    for k in list(set(voxk)):

        keep = voxk == k
        angi = bia[keep]
        median = np.median(angi)

        h_, x_ = np.histogram(lia[keep], bins=bins, weights=weights[keep], density=True)
        thetaLq_ = (x_[:-1]+x_[1:])/2
        alpha_ = [np.cos(np.radians(i))/Gtheta(i, thetaLq_, h_) for i in range(90)]
        # print('k, height density: ', h_)

        if show:
            plt.subplot(1,1,1)
            if k == 0 or k == list(set(voxk))[-1]:
                label = 'k=%i' %(k)
            else:
                label = None
            plt.plot(np.arange(0, 90, 1), alpha_, lw=0.5, color=colors[k], label=label)
            if k == list(set(voxk))[-1]: 
                plt.plot(np.arange(0, 90, 1), alpha, lw=3, ls='--', color='k', label='all')
                plt.axvline(57.5, ls='--', lw=3, color='r', label=r'$\theta_{0}=57.5$')
                # plt.axvline(scan_inc if scan_inc <= 90 else 180-scan_inc, ls='--', lw=3, color='orange', label=r'$\theta_{S}=%i$' %(scan_inc if scan_inc <= 90 else 180-scan_inc))
                plt.fill_between(ba, [alpha_f(i) for i in ba], color='yellow', alpha=0.5)
            plt.legend()
        
        # print('k, median = ', k, median)
        alpha_min = np.cos(np.radians(angi.min()))/Gtheta(angi.min(), thetaLq, h)
        alpha_max = np.cos(np.radians(angi.max()))/Gtheta(angi.max(), thetaLq, h)
        # alpha_median = np.cos(np.radians(ba[k]))/Gtheta(ba[k], thetaLq, h)
        alpha_median = np.cos(np.radians(median))/Gtheta(median, thetaLq, h)
        res.append([k, angi.min(), alpha_min, angi.max(), alpha_max, median, alpha_median])
        # print('k, median, alpha_median', k, median, alpha_median)

    if show:
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$alpha(\theta)$')

        savefig = os.path.join(resdir, 'alphas.png')
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

    return np.array(res)

def get_LADS_mesh(meshfile, voxel_size, kbins, kmax):

    mesh = o3d.io.read_triangle_mesh(meshfile)

    angles_mesh = true_angles(meshfile)
    angles_mesh = np.array(correct_angs(angles_mesh))
    # angles_mesh = np.array(angles_mesh)
    voxk = np.array(get_voxk_mesh(meshfile, voxel_size=voxel_size))
    # get surface area
    sa = mesh.get_surface_area()
    # Area per triangle
    area = np.full(len(voxk), np.round(sa/len(voxk), 6))

    # for volume
    vert = np.asarray(mesh.vertices)
    pcd = points2pcd(vert)
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    width, height, depth = voxel.get_max_bound() - voxel.get_min_bound()

    ar = np.arange(0, kmax, kbins)
    kcoords = []
    lads = []

    for i in range(len(ar)-1):
        kcoords.append([ar[i],ar[1:][i]-1])

    kcoords.append([ar[-1], kmax-1])
    # print(set(voxk))
    lads = []

    for i in kcoords:

        ki, kf = i
        
        keep = np.logical_and(voxk >= ki, voxk <= kf)
        
        # Area per leaf corrected by its angle with respect to the zenith
        Aleaf = area[keep]*np.cos(np.radians(angles_mesh[keep]))
        # plt.hist(np.cos(np.radians(angles_mesh[keep])), 30, label=i)
        # plt.legend()
        # plt.show()
    
        # Aleaf = area[keep]
        # Total area in bin
        A = Aleaf.sum()
        # get volume
        volume = width * height * kbins * voxel_size
        # print(volume)

        #save lad
        deltaH = (kf - ki)/2
        lads.append([(i[0]+deltaH)*voxel_size, A/volume])
        # print(kf, ki)
        # print(len(i), i[0], deltaH, A/volume)

    return np.array(lads)

def laod_scan_pos(filename):

    scan = np.loadtxt(filename, delimiter=',', dtype={'names':('scan', 'x', 'y', 'z'),
                'formats':('S4', 'f4', 'f4', 'f4')})

    return scan

def line2points_cartesian(p1, p2, res):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    l, m, n = x2-x1, y2-y1, z2-z1

    x = np.arange(x1, x2, res)
    if l == 0:
        print('l=0', p1, p2)
        z = z1
        y = (m/n) * (z - z1) + y1
    elif m == 0:
        print('m=0', p1, p2)
        y = y1
        z = (n/l) * (x - x1) + z1
    elif n == 0:
        print('n=0', p1, p2)
        y = (m/l) * (x - x1) + y1
        z = z1
    else:
        y = (m/l) * (x - x1) + y1
        z = (n/m) * (y - y1) + z1

    return x, y, z

def line2points_vect(p1, p2, res):

    v = np.array(p2) - np.array(p1)
    uv = v / np.linalg.norm(v)
    norm = np.linalg.norm(v)
    # proyect unitary vector to actual distance of beam and invert it to point cloud data view
    N = int(norm/res)
    pb = 1 * uv * np.linspace(0.01, norm, N)[:, np.newaxis] # this traces the beam with N points
    pp = np.array(p1) + pb
    # px, py, pz = pp.T[0], pp.T[1], pp.T[2]

    return pp


    
    


