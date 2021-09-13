import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
from scipy.stats import chisquare


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


def angs_dist(angs, true_angles=None, ws=None, downsample_debug=None, savefig=None, text=None):

    fig = plt.figure(figsize=(8,5))

    bins = np.linspace(0, 90, int(90/2))
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
        textres = '$\chi^2=%.4f$ \n ds_debug=%i \n weights=%s' %(chi2, downsample_debug, ws.any())
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
        return np.array(x[0]), np.array(x_truth[0])
    else:
        return np.array(x[0])



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
    