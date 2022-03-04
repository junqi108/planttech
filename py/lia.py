import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys, glob
import laspy as lp
from scipy.stats import chisquare
from time import process_time

import loads
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

def vecttoangle(v1, v2):
    
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    
    return np.rad2deg(angle)

def mesh_leaf_area(meshfile):
    '''
    Get the leaf area from mesh file.
    '''

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

def get_voxk(points, voxel_size=0.5, mesh=False):

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

def get_normals(points, kd3_sr, max_nn, show=False, downsample=False):
    
    pcd = loads.points2pcd(points)
    
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=kd3_sr,max_nn=max_nn))
    
    if downsample:
        
        downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.01)

        if show:
            o3d.visualization.draw_geometries([downpcd])
        
        return np.asarray(downpcd.normals), np.asarray(downpcd.points)

    else:
        
        return np.asarray(pcd.normals)

def get_leaf_angle(normals):

    angs = {'avgAngle':[]}

    for normal in normals:

        ang = vecttoangle([0, 0, 1], -normal)
        angs['avgAngle'].append(np.round(ang, 2))

    return angs

def correct_angs(angs):

    thetaL = [i if (i < 90) else (180 - i) for i in angs]

    return thetaL

def get_weigths(points, voxel_size=0.5):

    pcd = loads.points2pcd(points)
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
    # print(voxel_size, np.sum(density == 0))
    # Mean volume density
    mean_density = np.median(density)
    # print('Mean density: \t', mean_density)
    # Weigth value per voxel
    w = density / mean_density
    # Weigth value per point
    ws = w[idxinv]

    return ws

def leaf_angle(points, mockname, treename, voxel_size_w, kd3_sr, max_nn, save=False,
savefig=False, text=None, downsample=False, weigths=True, voxel_size_h=1, ismock=True,
ylim=None, ylimh=None, savefig_extra=False):
    
    mockdir = os.path.join(_data, mockname)
    resdir = os.path.join(mockdir, 'lia')
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    # Mesh file
    if ismock:
        meshfile = os.path.join(mockdir, 'mesh.ply')
        if os.path.isfile(meshfile):
            ta = true_angles(meshfile)
            voxk_mesh = get_voxk_mesh(meshfile, voxel_size=voxel_size_h)
        else:
            raise ValueError('No mesh.ply file in %s' %(mockdir))
    else:
        ta = None
        voxk_mesh = None

    # Compute the normals to fpc (nfpc)
    if downsample:
        normals, points = get_normals(points, kd3_sr, max_nn, show=False, downsample=downsample)
    else:
        normals = get_normals(points, kd3_sr, max_nn, show=False, downsample=downsample)

    # Get the Leaf Inclination Angle
    angs = get_leaf_angle(normals)
    thetaL = angs['avgAngle']

    thetaL = correct_angs(angs['avgAngle'])
    if ismock:
        ta = correct_angs(ta)

    if savefig:
        _savefig = os.path.join(resdir, 'leaf_angle_dist_%s.png' %(treename))
        _savefig_k = os.path.join(resdir, 'leaf_angle_dist_height_%s.png' %(treename))
    elif savefig_extra:
        _savefig = os.path.join(resdir, 'leaf_angle_dist_%s_%.3f_%.3f_%i.png' %(treename, voxel_size_w, kd3_sr, max_nn))
        _savefig_k = os.path.join(resdir, 'leaf_angle_dist_height_%s_%.3f_%.3f_%i.png' %(treename, voxel_size_w, kd3_sr, max_nn))
    else:
        _savefig = None
        _savefig_k = None

    if weigths:
        ws = get_weigths(points, voxel_size=voxel_size_w)
    else:
        ws = None

    if ismock:
        h, thetaLq, htruth = figures.angs_dist(thetaL, ta, savefig=_savefig, text=text, ws=ws)
    else:
        h, thetaLq = figures.angs_dist(thetaL, ta, savefig=_savefig, text=text, ws=ws, ylim=ylim)

    voxk = get_voxk(points, voxel_size=voxel_size_h)

    minpoint = points[:,2].min()
    freq, kbin = figures.angs_dist_k(voxk, voxk_mesh, thetaL, ta, ws=ws, savefig=_savefig_k, ylim=ylimh, minpoint=minpoint, voxel_size_h=voxel_size_h)
    
    height_dist_dict = {'zmin':np.round(points[:,2].min() + (kbin[:-1] * voxel_size_h), 2), 
                        'zmax':np.round(points[:,2].min() + (kbin[1:] * voxel_size_h), 2), 
                        'freq':np.round(freq, 4)}
    values_h = [points[:,2][np.logical_and(points[:,2] >= i, points[:,2] < j)] for i,j in zip(height_dist_dict['zmin'], height_dist_dict['zmax'])]
    height_dist_dict['n'] = [len(i) for i in values_h]
    # print(height_dist_dict['n'])
    height_dist_dict['values'] = [np.random.choice(i, size=100) if len(i) >= 100 else i for i in values_h]
    # values_randoms = []
    # for i in values_h:
    #     if len(i) <= 100: values_randoms.append(i)
    #     elif len(i) > 0: values_randoms.append([0])
    #     else: np.random.choice(i, size=100)

    angs_dist_dict = {'angle_min':np.round(thetaLq[:-1], 1), 
                      'angle_max':np.round(thetaLq[1:], 1), 
                      'freq':np.round(h,4)}
    values = [np.array(thetaL)[np.logical_and(thetaL >= i, thetaL < j)] for i,j in zip(angs_dist_dict['angle_min'], angs_dist_dict['angle_max'])]
    angs_dist_dict['n'] = [len(i) for i in values]
    angs_dist_dict['values'] = [np.random.choice(i, size=100) if len(i) >= 100 else i for i in values]

    # print(pd.DataFrame.from_dict(height_dist_dict))
    # Save angles and weights
    if save:
        outdir_angs = os.path.join(resdir, 'angles_%s.npy' %(treename))
        np.save(outdir_angs, thetaL)

        # save angle and weighted frequencies
        outdir_angs_freq = os.path.join(resdir, 'angles_freq_%s.csv' %(treename))
        pd.DataFrame.from_dict(angs_dist_dict).to_csv(outdir_angs_freq, index=False)

        # save height and weighted frequencies
        outdir_height_freq = os.path.join(resdir, 'height_freq_%s.csv' %(treename))
        pd.DataFrame.from_dict(height_dist_dict).to_csv(outdir_height_freq, index=False)

        if weigths:
            outdir_ws = os.path.join(resdir, 'weights_%s.npy' %(treename))
            np.save(outdir_ws, ws)

        # print('angles:', np.hist())
        # print('weights:', len(ws))

    if ismock:

        if float(0) in htruth:
            chi2, p = chisquare(h+1, htruth+1)
        else:
            chi2, p = chisquare(h, htruth)

        return chi2

def bestfit_pars_la(points, mockname, treename, weigths=True, ismock=True):
    '''
    Find the best fit parameters comparing chi2 of estimated LIA with Truth.
    '''

    mockdir = os.path.join(_data, mockname)
    resdir = os.path.join(mockdir, 'lia')
    outfile = os.path.join(resdir, 'bestfit_%s.npy' %(treename))

    if not os.path.exists(resdir):
        os.makedirs(resdir)

    voxel_size_w = 0.1
    kd3_sr = 1.0
    max_nn = 10

    pars = {}
    pars['voxel_size_w'] = [0.0001, 0.001, 0.01, 0.1, 1]
    pars['kd3_sr'] = [0.001, 0.01, 0.1, 1.0]
    pars['max_nn'] = [3, 5, 10, 20, 50, 100]

    # Mesh file
    meshfile = os.path.join(mockdir, 'mesh.ply')

    if ismock:
        if os.path.isfile(meshfile):
            la = mesh_leaf_area(meshfile)
        else:
            raise ValueError('No mesh.ply file in %s' %(mockdir))
    else:
        la = 0.

    res = {}
    res['leafsize'] = la
    res['voxel_size_w_fixed'] = voxel_size_w
    res['kd3_sr_fixed'] = kd3_sr
    res['max_nn_fixed'] = max_nn

    for key, val in pars.items():

        res[key] = []

        for par in val:

            if key == 'voxel_size_w':
                chi2 = leaf_angle(points, mockname, treename, 
                par, kd3_sr, max_nn, weigths=weigths, ismock=ismock)
            elif key == 'kd3_sr':
                chi2 = leaf_angle(points, mockname, treename, 
                voxel_size_w, par, max_nn, weigths=weigths, ismock=ismock)
            elif key == 'max_nn':
                chi2 = leaf_angle(points, mockname, treename, 
                voxel_size_w, kd3_sr, par, weigths=weigths, ismock=ismock)
            else:
                raise ValueError('%s is not a valid parameter' %(key))

            res[key].append([treename, par, chi2])
            print(key, par, 'DONE...')

        df = pd.DataFrame(res[key], columns=['tree', 'value', 'chi2'])
        keep = (df['chi2'] == df['chi2'].min())
        bestfit_par = df.loc[keep, 'value'].values[0]

        res[key+'_'+'bestfit'] = bestfit_par
        print(key, 'BESTFIT:\t', bestfit_par)

    # save dict
    np.save(outfile, res)

    return res

def best_fit_pars_plot(res, treename, mockname, savefig=None):

    pars = ['voxel_size_w', 'kd3_sr', 'max_nn']

    fig = plt.figure(figsize=(15, 4*len(pars)))

    for num, par in enumerate(pars):

        plt.subplot(len(pars),1,num+1)

        x = np.array(res[par])[:,1].astype('float')
        y = np.array(res[par])[:,2].astype('float')

        if par in ['voxel_size_w', 'kd3_sr']:
            plt.semilogx(x, y, lw=3, marker='*')
        else:
            plt.plot(x, y, lw=3, marker='*')

        plt.axvline(res[par+'_'+'bestfit'], lw=2, ls='--', color='g')
        
        plt.xlabel(par)
        plt.ylabel(r'$\chi^2$')

    if savefig is None:

        mockdir = os.path.join(_data, mockname)
        savefig = os.path.join(mockdir, 'lia', 'bestfits_pars_%s.png' %(treename))

    plt.savefig(savefig, dpi=200, bbox_inches='tight')
