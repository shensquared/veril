import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import samples


def plot_funnel(V, sys_name):
    x = samples.levelsetData(V)[0]
    plt.fill(x[:, 0], x[:, 1])
    if sys_name is 'VanderPol':
        xlim = np.load('../data/VanderPol/VanDerPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='ROA boundary')
    plt.show()


def scatterSamples(samples, sys_name):
    plt.scatter(samples[:, 0], samples[:, 1])
    if sys_name is 'VanderPol':
        xlim = np.load('../data/VanderPol/VanDerPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='ROA boundary')
    plt.show()
