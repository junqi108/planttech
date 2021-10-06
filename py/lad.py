import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys, glob
import laspy as lp
from scipy.stats import chisquare
from time import process_time

import figures
import loads
import lia

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

def get_meshfile(mockname):

    mockdir = os.path.join(_data, mockname)
    meshfile = os.path.join(mockdir, 'mesh.ply')
    if os.path.isfile(meshfile):
        return meshfile
    else:
        raise ValueError('No mesh.ply file in %s' %(mockdir))

def within_bounds(voxel_idx, voxel_idx_bounds):
    
    vdict = idx_bounds(voxel_idx_bounds, False)
    
    keep = np.ones(voxel_idx.shape[0], dtype=bool)
    for i in range(3):
        keep &= np.logical_and(vdict['ijk_min'][i] <= voxel_idx[:,i], vdict['ijk_max'][i] >= voxel_idx[:,i])
        
    # Check 3D matrix i,j,k size matches length of solid voxel grid index array: i*j*k == voxs_idx.shape[0]
    vdict = idx_bounds(voxel_idx[keep], False)
    assert np.product(np.array(vdict['ijk_max'])+1) == voxel_idx_bounds.shape[0]
        
    return voxel_idx[keep]

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

def get_voxk(points, voxel_size=0.5):

    pcd = loads.points2pcd(points)
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

def downsample_lia(mockname, treename, inds):

    # Check if angles and weights are available
    outdir_angs = os.path.join(_data, mockname, 'lia', 'angles_%s.npy' %(treename))
    outdir_ws = os.path.join(_data, mockname, 'lia', 'weights_%s.npy' %(treename))

    lia = np.load(outdir_angs)
    ws = np.load(outdir_ws)

    return lia[inds], ws[inds]

def vecttoangle(v1, v2):
    
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    
    return np.rad2deg(angle)

def get_bia(points, sensors):

    # get beam inclination angle (BIA)
    bia = []
    for i, j in zip(points, sensors):
        # beam vector
        v = np.array(j) - np.array(i)
        # beam unitary vector
        uv = v / np.linalg.norm(v)
        bia.append(vecttoangle([0, 0, 1], -uv))

    return bia

def compute_attributes(points, resdir, voxel_size, treename):

    # get points voxel bounding box
    pcd = loads.points2pcd(points)
    voxp = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # Create voxel of plant region
    width, height, depth = voxp.get_max_bound() - voxp.get_min_bound()
    # print(width, height, depth)
    voxs = o3d.geometry.VoxelGrid.create_dense(origin=voxp.origin, color=np.array([0,0,1]), voxel_size=voxel_size, width=width, height=height, depth=depth)

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
    # Check if attributes are available for this tree
    attributes_file = os.path.join(resdir, 'm3s_%s_%s.npy' %(treename, str(voxel_size)))
    if os.path.isfile(attributes_file):
        m3b = np.load(attributes_file)
    else:
        print('No ray tracing for tree %s and voxel size %.3f' %(treename, voxel_size))
        m3b = np.ones(np.array(vdict['ijk_max'])+1, dtype=bool)
        
    print('foliage voxel dimensions: \t', m3s.shape)
    print('ray tracker voxel dimensions: \t', m3b.shape)

    # Check 3D matrix i,j,k size matches length of solid voxel grid index array: i*j*k == voxs_idx.shape[0]
    assert np.product(np.array(vdict['ijk_max'])+1) == voxs_idx.shape[0]

    # get voxel grid indexes for points and beams voxels
    voxp_idx = get_voxels(voxp)

    # get rid of voxels outside the plant region
    voxp_idx = within_bounds(voxp_idx, voxs_idx)

    # fill 3D matrix with True if voxel exist
    m3p = m3s.copy()
    for (i,j,k) in voxp_idx:
        
        m3p[i][j][k] = True
        
    print('Number of voxels ocupied by points cloud: \t %i' %(m3p.sum())) 
    print('Number of voxels ocupied by beam points cloud: \t %i' %(m3b.sum()))
    print('Total number of voxels in plant regions: \t %i' %((~m3s).sum()))

    m3att = get_attributes(m3s.shape, m3p, m3b, True)

    return m3att

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

def alpha_k(bia, voxk, lias, ws, resdir, meshfile, figext=None, klia=False, use_true_lia=False):

    colors = plt.cm.jet(np.linspace(0,1,len(set(voxk))))
    # uv = beam_direction(beam_angles.T[0], beam_angles.T[1], scan_inc)
    
    # bins = np.linspace(0, 90, int(90/1)) # Don't change this!!!
    bins = np.linspace(0, 90, 90) # Don't change this!!!

    weights = 1/ws

    if use_true_lia:
        ta = lia.true_angles(meshfile)
        ta = lia.correct_angs(ta)
        h, x = np.histogram(ta, bins=bins, density=True)
    else:
        h, x = np.histogram(lias, bins=bins, weights=weights, density=True)
    
    thetaLq = (x[:-1]+x[1:])/2
    alpha = [np.cos(np.radians(i))/Gtheta(i, thetaLq, h) for i in bins]
    alpha_f = lambda theta: np.cos(np.radians(theta))/Gtheta(theta, thetaLq, h)

    bia_ = bia
    bia = np.array(lia.correct_angs(np.array(bia_)))
    # bia = np.array(np.abs(bia_))
    bamin, bamax = np.percentile(bia, (0.3,99.7))
    ba = np.linspace(bamin, bamax, len(set(voxk)))

    if figext is not None:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(1,1,1)
        plt.hist(np.array(bia_), 40, histtype='step', label='pre correction')
        plt.hist(bia, 40, histtype='step', label='after correction')
        plt.legend()
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$Frecuency$')

        savefig = os.path.join(resdir, 'figures','bia_%s.png' %(figext))
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

    if figext is not None:
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
        savefig = os.path.join(resdir, 'figures', 'bia_per_k_%s.png' %(figext))
        plt.savefig(savefig, dpi=200, bbox_inches='tight')


    res = []
    if figext is not None:
        fig = plt.figure(figsize=(14, 6))

    for k in list(set(voxk)):

        keep = voxk == k
        angi = bia[keep]
        median = np.median(angi)

        h_, x_ = np.histogram(lias[keep], bins=bins, weights=weights[keep], density=True)
        thetaLq_ = (x_[:-1]+x_[1:])/2
        alpha_ = [np.cos(np.radians(i))/Gtheta(i, thetaLq_, h_) for i in range(90)]
        # print('k, height density: ', h_)

        if figext is not None:
            plt.subplot(1,1,1)
            if k == 0 or k == list(set(voxk))[-1]:
                label = 'k=%i' %(k)
            else:
                label = None
            plt.plot(np.arange(0, 90, 1), alpha_, lw=0.5, color=colors[k], label=label)
            if k == list(set(voxk))[-1]: 
                plt.plot(bins, alpha, lw=3, ls='--', color='k', label='all')
                plt.axvline(57.5, ls='--', lw=3, color='r', label=r'$\theta_{0}=57.5$')
                # plt.axvline(scan_inc if scan_inc <= 90 else 180-scan_inc, ls='--', lw=3, color='orange', label=r'$\theta_{S}=%i$' %(scan_inc if scan_inc <= 90 else 180-scan_inc))
                plt.fill_between(ba, [alpha_f(i) for i in ba], color='yellow', alpha=0.5)
            plt.legend()
        
        if klia:
            T, H = thetaLq_, h_
        else:
            T, H = thetaLq, h

        alpha_min = np.cos(np.radians(angi.min()))/Gtheta(angi.min(), T, H)
        alpha_max = np.cos(np.radians(angi.max()))/Gtheta(angi.max(), T, H)
        alpha_median = np.cos(np.radians(median))/Gtheta(median, T, H)

        # add mean weights per k
        mean_weights_k = np.mean(weights[keep])
        res.append([k, angi.min(), alpha_min, angi.max(), alpha_max, median, alpha_median, mean_weights_k])
        # print('k, median, alpha_median', k, median, alpha_median)

    if figext is not None:
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$alpha(\theta)$')

        savefig = os.path.join(resdir, 'figures', 'alphas_%s.png' %(figext))
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

    return np.array(res)

def get_lad_perk(kcoord, m3att, alphas, voxel_size, alpha2, mean_weights):
    
    ki, kf = kcoord
    # print(kf-ki)

    if mean_weights is not None:
        betas = mean_weights
    else:
        betas = alphas*0 + 1
    
    if kf > m3att.shape[2]:
        raise ValueError('k values cannot be greater than available. Maximum K value is: %i' %(m3att.shape[2]))
    
    m3 = m3att[:,:,ki:kf]
    DeltaH = (kf-ki) * voxel_size
    lai = []
    
    for i in range(kf-ki):
        
        nI = (m3[:,:,i] == 1).sum()
        nP = (m3[:,:,i] == 2).sum()
        n0 = (m3[:,:,i] == 3).sum()
        _lai = nI/(nI+nP)
        # _lai = nI/(nI+nP+n0)
        alpha = alphas[i]
        beta = betas[i]
        # print(1/DeltaH, alpha, _lai)
        # lai.append(_lai)
        lai.append(alpha * beta * _lai)
#         print(i, nI, nP, nI/(nI+nP))
#         print(i, nI/(nI+nP), DeltaH)

    # if mean_weights is not None:
    #     beta = np.mean(mean_weights)
    # else:
    #     beta = 1
        
    # LAD = alpha2 * np.mean(alphas) * (1/DeltaH) * np.array(lai).sum()
    LAD = alpha2 * (1/DeltaH) * np.array(lai).sum()

#     print(alpha, 1/DeltaH, np.array(lai).sum(), LAD)
    # print('k, mean alphas: ', kcoord, np.mean(alphas))
    
    if mean_weights is not None:
        print((ki+(kf-ki-1)/2)*voxel_size, betas)

    return (ki+(kf-ki-1)/2)*voxel_size, LAD
    
#
def get_LADS(m3att, voxel_size, kbins, alphas_k, alpha2, mean_weights_k=None):
    
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
        if mean_weights_k is not None:
            mean_weights = mean_weights_k[ki:kf]
        else:
            mean_weights = None
        # print(i, alphas)
        h, lad = get_lad_perk(i, m3att, alphas, voxel_size, alpha2, mean_weights)
        lads.append([h, lad])
        
    return np.array(lads)

def get_LADS_mesh(meshfile, voxel_size, kbins, kmax):

    mesh = o3d.io.read_triangle_mesh(meshfile)

    angles_mesh = lia.true_angles(meshfile)
    angles_mesh = np.array(lia.correct_angs(angles_mesh))
    # angles_mesh = np.array(angles_mesh)
    voxk = np.array(get_voxk_mesh(meshfile, voxel_size=voxel_size))
    # get surface area
    sa = mesh.get_surface_area()
    # Area per triangle
    area = np.full(len(voxk), np.round(sa/len(voxk), 6))

    # for volume
    vert = np.asarray(mesh.vertices)
    pcd = loads.points2pcd(vert)
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