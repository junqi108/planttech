import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys, glob
import laspy as lp
from dbfread import DBF
from scipy.stats import chisquare
from time import process_time
from scipy import interpolate
import pyrr 
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'Omar A. Ruiz Macias'
__copyright__ = 'Copyright 2021, PLANTTECH'
__version__ = '0.1.0'
__maintainer__ = 'Omar A. Ruiz Macias'
__email__ = 'omar.ruiz.macias@gmail.com'
__status__ = 'Dev'

# Global
basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_py = os.path.join(basedir, 'py')
_data = os.path.join(basedir, 'data')
_images = os.path.join(basedir, 'images')

def numpy2npy(mockname, downsample=None):
    '''
    Convert Blensor output txt files that have fake numpy
    extension to real npy.
    '''

    datapath = os.path.join(_data, mockname)

    # read the numpy files
    for file in glob.glob(os.path.join(datapath, 's*.numpy')):
        df = np.loadtxt(file)
        filename = file.split('/')[-1].split('.')[0]
        print('%s done --> Number of beams: %i' %(file.split('/')[-1], len(df)))
        outfile = os.path.join(_data, mockname, filename)

        if downsample is not None:
            keep = np.random.randint(0, len(df), int(len(df) * downsample))
            # print(filename, len(df), len(df[keep]))
            np.save(outfile, df[keep])
        else:
            np.save(outfile, df)

def load_scan_pos(filename):
    '''
    Read file with sensor positions.
    '''

    scan = np.loadtxt(filename, delimiter=',', dtype={'names':('scan', 'x', 'y', 'z'),
                'formats':('S4', 'f4', 'f4', 'f4')})

    return scan

def npy2pandas(mockname, downsample=None):
    '''
    Read the npy files and merge in a pandas dataframe.
    This function also adds the information of the sensor
    coordinates called the bia (beam inclination angle) positions.
    '''

    datapath = os.path.join(_data, mockname)

    # read the numpy files
    files = {}
    bia_pos = {}
    N = len(glob.glob(os.path.join(datapath, 's*.npy')))
    print('Number of files: %i' %(N))
    for file in glob.glob(os.path.join(datapath, 's*.npy')):
        df = np.load(file)
        filename = file.split('/')[-1]

        if downsample is not None:
            keep = np.random.randint(0, len(df), int(len(df) * downsample))
            df = df[keep]

        # sensor coordinates
        spos = os.path.join(datapath, 'scanner_pos.txt')
        scan = load_scan_pos(spos)
        id = [i.decode("utf-8") for i in scan['scan']]
        keep = np.array(id) == filename[:3]
        _, sx, sy, sz = scan[keep][0]

        if df.shape[0] > 0:
            files[filename] = df
            bia = np.full((len(df),3), np.array([sx, sy, sz]))
            bia_pos[filename] = bia
            # print(filename, df.shape, bia.shape)
        else:
            print('file %s empty' %(filename))

    # concatenate all data
    # if len(list(files.keys())) < 2:
        # print(list(files.values()))
    #     df = list(files.values())[0]
    #     bias = list(bia_pos.values())[0]
    # else:
    df = np.concatenate(list(files.values()))
    bias = np.concatenate(list(bia_pos.values()))

    # pass this to a pandas data frame for simplicity
    scan = pd.DataFrame(df, columns=['timestamp', 'yaw', 'pitch', 'distance','distance_noise',
                                    'x','y','z',
                                    'x_noise','y_noise','z_noise',
                                    'object_id', 'color0', 'color1','color2', 'idx'])

    bia = pd.DataFrame(bias, columns=['sx', 'sy', 'sz'])

    # concat
    scan = pd.concat((scan, bia), axis=1)

    return scan

def points2pcd(points, colors=None):
    '''
    Numpy 3D-array to open3D PCD.
    '''
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:

        if isinstance(colors, list):
            color = np.full_like(points, colors)
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=np.asarray(pcd.points).shape))
    
    return pcd

def DF2array(df):

    # points = np.vstack((np.array(df['x']), np.array(df['y']), np.array(df['z']))).transpose()
    points = df.to_numpy()

    return points
    
def showPCfromDF(df):

    points = np.vstack((np.array(df['x']), np.array(df['y']), np.array(df['z']))).transpose()
    pcd = points2pcd(points)
    o3d.visualization.draw_geometries([pcd])


def extract_leaves(df, show=False):
    '''
    Extract leaves from dataframe.
    '''

    leaves = df['object_id'] == 1986094444.0
    if show:
        showPCfromDF(df[leaves])

    return leaves


def loadlaz(name):

    filepath = os.path.join(_data, name, 'lidar.laz')
    las = lp.read(filepath)

    return las

def loaddbf(name):

    filepath = os.path.join(_data, name, 'trajectory.dbf')
    table = DBF(filepath, load=True)

    return table

def showPCDS(pointslist, colours):

    pcds = []

    for num, points in enumerate(pointslist):

        pcd = points2pcd(points)

        if isinstance(colours[num], list):
            color = np.full_like(points, colours[num])
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(colours[num])

        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)

def interp_traj(traj, gpstime):

    tck, u = interpolate.splprep([np.array(traj.x), np.array(traj.y), np.array(traj.z)], s=2)
    limits = traj['gpstime'].min(), traj['gpstime'].max()
    gpstimenorm = normgpstime(gpstime, limits)
    xs, ys, zs = interpolate.splev(gpstimenorm, tck)

    return xs, ys, zs

def normgpstime(gpstime, limits):
    '''
    Normalize the gpstime to run from zero to one.
    '''

    dd = limits[1] - limits[0]
    gpstimenorm = (gpstime - limits[0]) / dd

    return gpstimenorm

def coordsDF(las, traj):

    xs, ys, zs = interp_traj(traj, las.gps_time)
    df = np.vstack((las.x, las.y, las.z, xs, ys, zs)).transpose()
    df = pd.DataFrame(df, columns=['x', 'y', 'z', 'xs', 'ys', 'zs'])

    return df

def showbeams(df):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = DF2array(df[['x', 'y', 'z']])
    sensors = DF2array(df[['xs', 'ys', 'zs']])

    for p1, p2 in zip(points, sensors):

        line = pyrr.line.create_from_points(p1, p2, dtype=None)
        ax.plot(*line.T.tolist(), lw=0.1)
        ax.scatter3D(*p1, c='g', s=2)
        ax.scatter3D(*p2, c='r', s=5)

def remove_outliers(a, b, nbins=100, bounds=(1, 99)):

    bins = np.linspace(a.min(), a.max(), nbins)
    keep = np.zeros(len(a), dtype=bool)
    res = []

    for i in range(len(bins)-1):
        
        mask = (a > bins[i]) & (a < bins[i+1])
        median = np.median(b[mask])
        low, upp = np.percentile(b[mask], bounds)
        mask &= (b > low) & (b < upp)
        res.append([(bins[i+1] + bins[i])/2, median, low, upp])

        plt.scatter(a[mask], b[mask], s=0.01)

        keep |= mask

    return res, keep

def load_lias_ws(mockname, treename):

    # Check if angles and weights are available
    outdir_angs = os.path.join(_data, mockname, 'lia', 'angles_%s.npy' %(treename))
    outdir_ws = os.path.join(_data, mockname, 'lia', 'weights_%s.npy' %(treename))

    lia = np.load(outdir_angs)
    ws = np.load(outdir_ws)

    return lia, ws


