
import os, sys

basedir = os.path.dirname(os.getcwd())
_py = os.path.join(basedir, 'py')

sys.path.insert(1, _py)
import ray as rayt

if __name__ == "__main__":

    voxel_size = 0.13
    downsample = None
    sample = None
    show = False
    stop = None
    mockname = 'single_57_combined_lite'

    m3s = rayt.main(mockname, voxel_size, downsample=downsample, sample=sample, stop=stop, show=show)
