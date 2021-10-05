import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
from scipy.stats import chisquare
import os, glob

import lad

basedir = os.path.dirname(os.getcwd())
_data = os.path.join(basedir, 'data')


colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB',
                 '#ee7111', '#0ce5f4', '#9d0eb2',
                 '#bbfb00', '#00fba4', '#fb00e0'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray', labelsize=12)
plt.rc('ytick', direction='out', color='gray', labelsize=12)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=12)    # legend fontsize


def angs_dist(angs, true_angles=None, ws=None, savefig=None, text=None):

    fig = plt.figure(figsize=(8,5))

    bins = np.linspace(0, 90, int(90/5))
    # bins = np.linspace(0, 180, int(180/4))

    if ws is None:
        weights = None
    else:
        weights = 1/ws

    x = plt.hist(angs, bins=bins, weights=weights, histtype='step', lw=3, color='g', density=True, label='Estimated')
    if true_angles is not None:
        x_truth = plt.hist(true_angles, bins=bins, histtype='step', lw=2,  ls='--', color='r', density=True, label='Truth')
        plt.legend()

        if float(0) in np.array(x_truth[0]):
            chi2, p = chisquare(np.array(x[0])+1, np.array(x_truth[0])+1)
        else:
            chi2, p = chisquare(np.array(x[0]), np.array(x_truth[0]))

        maxh = np.array(x[0]).max()
        box = dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.3)
        textres = '$\chi^2=%.4f$ \n weights=%s' %(chi2, ws.any())
        plt.text(40, maxh-maxh/5, textres, size=12, bbox=box)

    if text is not None:
        maxh = np.array(x[0]).max()
        box = dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.3)
        plt.text(0, maxh-maxh/5, text, size=12, bbox=box)

    # plt.axvline(35, c='r', label=r'$\theta_{L}=35$')
    # plt.axvline(180-35, c='r', ls='--', label=r'$\theta_{L}=180-35$')

    plt.ylabel(r'Frecuency')
    plt.xlabel(r'$\theta_{L}$')

    # 
    plt.ioff()

    if savefig is not None:
        plt.savefig(savefig, dpi=200, bbox_inches='tight')
    
    if true_angles is not None:
        return np.array(x[0]), np.array(x[1]), np.array(x_truth[0])
    else:
        return np.array(x[0]), np.array(x[1])



def angs_dist_k(voxk, voxk_mesh, angs, true_angles, ws=None, savefig=None):

    fig = plt.figure(figsize=(8,5*2))
    angs = np.array(angs)
    true_angles = np.array(true_angles)

    bins = np.linspace(0, 90, int(90/5))
    bins_k = list(set(voxk_mesh))
    bins_ = np.linspace(0, len(bins_k), len(bins_k)+1)

    if ws is None:
        weights = None
    else:
        weights = 1/ws

    plt.subplot(2, 1, 1)
    plt.title('Height Normalized')
    plt.hist(voxk, bins=bins_, weights=weights, align='left', histtype='step', lw=3, color='g', density=True, label='Estimated')
    plt.hist(voxk_mesh, bins=bins_, align='left', histtype='step', lw=2,  ls='--', color='r', density=True, label='Truth')
    plt.legend()
    plt.ylabel(r'Frecuency')
    plt.xlabel(r'Height (voxel K)')

    plt.subplot(2, 1, 2)

    plt.axhline(0, ls='--', lw=2, color='k')

    for k in bins_k:

        keep = voxk == k
        keep_mesh = voxk_mesh == k
        if weights is None:
            hist, x = np.histogram(angs[keep], bins=bins, density=True)
        else:
            hist, x = np.histogram(angs[keep], bins=bins, weights=weights[keep], density=True)
        hist_mesh, _ = np.histogram(true_angles[keep_mesh], bins=bins, density=True)
        delta = hist_mesh - hist

        x = (x[:-1]+x[1:])/2
        plt.plot(x, delta, lw=1.5, label=k)

    plt.ylabel(r'Frecuency')
    plt.xlabel(r'$\theta_{L}$')
    plt.legend(fontsize=8)

    # 
    plt.ioff()

    if savefig is not None:
        plt.savefig(savefig, dpi=200, bbox_inches='tight')

def G_alpha_plot(G_alpha_theta, savefig=None):

    fig = plt.figure(figsize=(10,5))

    plt.plot(G_alpha_theta[0], G_alpha_theta[1], label=r'$G(\theta)$')
    plt.plot(G_alpha_theta[0], G_alpha_theta[2], label=r'$\alpha(\theta)$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$f(\theta)$')
    plt.legend(loc='lower left')

    # 
    plt.ioff()

    if savefig is not None:
        plt.savefig(savefig, dpi=200, bbox_inches='tight')
    
def show_beams(mockname, sample, tracers=False, res=1):

    # show sample of beams
    # colors = ['b', 'cyan', 'orange', 'yellow']
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    mockdir = os.path.join(_data, mockname)
    rawdata_files = glob.glob(os.path.join(mockdir, 's*.npy'))

    colors = plt.cm.jet(np.linspace(0,1,len(rawdata_files)))

    # Results directory
    resdir_name = '%s_%s' %('results', mockname)
    resdir = os.path.join(mockdir, resdir_name)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    # read the numpy files
    for num, file in enumerate(rawdata_files):

        df = np.load(file)
        idx = np.random.randint(0, len(df), sample, dtype=int)
        df = df[idx]

        filename = file.split('/')[-1]
        x, y, z = df.T[5], df.T[6], df.T[7]

        # get sensor coordinates
        try:
            spos = os.path.join(mockdir, 'scanner_pos.txt')
        except Exception as e:
            raise ValueError(e)

        scan = lad.laod_scan_pos(spos)
        id = [i.decode("utf-8") for i in scan['scan']]
        keep = np.array(id) == filename[:3]
        _, sx, sy, sz = scan[keep][0]

        # sensor coordinates
        p2 = [sx, sy, sz]
        idx = np.random.randint(0, len(x), 100, dtype=int)

        ax.scatter3D(x, y, z, c='g', s=1)
        ax.scatter3D(sx, sy, sz, c='r', s=10)

        # for each LiDAR point, draw a point-like line trhough the sensor
        for i,j,k in zip(x, y, z):
            
            p1 = [i, j, k]
            ax.plot3D([i, sx], [j, sy], [k, sz], lw=0.1, color=colors[num])

            if tracers:
                pp = lad.line2points_vect(p1, p2, res=res)
                px, py, pz = pp.T[0], pp.T[1], pp.T[2]
                ax.scatter3D(px, py, pz, c='k', s=1)