import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
from scipy.stats import chisquare


colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=14)    # legend fontsize


def angs_dist(angs, true_angles=None, ws=None, savefig=None, text=None):

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

        maxh = np.array(x_truth[0]).max()
        box = dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.3)
        plt.text(40, maxh, r'$\chi^2=%.4f$' %(chi2), size=12, bbox=box)

    if text is not None:
        maxh = np.array(x_truth[0]).max()
        box = dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.3)
        plt.text(0, maxh-maxh/10, text, size=12, bbox=box)

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