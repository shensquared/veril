import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import samples


def plot_funnel(V, sys_name, slice_idx=None):
    file_dir='../data/' + sys_name
    x = samples.levelsetData(V, slice_idx)[0]
    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :], color='red', label='ROAboundary')
    else:
        stable_samples = np.load(file_dir+'/stableSamplesSlice.npy')
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color = 'red')
    plt.fill(x[:, 0], x[:, 1])
    plt.show()


def scatterSamples(samples, sys_name):
    plt.scatter(samples[:, 0], samples[:, 1])
    if sys_name is 'VanderPol':
        xlim = np.load('../data/VanderPol/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='ROA boundary')
    plt.show()
