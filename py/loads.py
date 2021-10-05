import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys, glob
import laspy as lp
from scipy.stats import chisquare
from time import process_time

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
        outfile = os.path.join(_data, mockname, filename)

        if downsample is not None:
            keep = np.random.randint(0, len(df), int(len(df) * downsample))
            print(filename, len(df), len(df[keep]))
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
            print(filename, df.shape, bia.shape)
        else:
            print('file %s empty' %(filename))

    # concatenate all data
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

def points2pcd(points):
    '''
    Numpy 3D-array to open3D PCD.
    '''
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=np.asarray(pcd.points).shape))
    
    return pcd

def DF2array(df):

    points = np.vstack((np.array(df['x']), np.array(df['y']), np.array(df['z']))).transpose()

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

