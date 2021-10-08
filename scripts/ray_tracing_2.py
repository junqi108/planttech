
import os, sys

basedir = os.path.dirname(os.getcwd())
_py = os.path.join(basedir, 'py')

sys.path.insert(1, _py)
import ray as rayt

if __name__ == "__main__":

    # voxel_size = 0.
    downsample = None
    sample = None
    show = False
    stop = None
    mockname = 'single_combined_lite'

    for i in [0.35, 0.4]:
        voxel_size = i
        m3s, m3count = rayt.main(mockname, voxel_size, downsample=downsample, sample=sample, stop=stop, show=show)
