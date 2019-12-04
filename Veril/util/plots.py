import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import samples


def plot_funnel(V):
    x = samples.levelsetData(V)[0]
    plt.fill(x[:, 0], x[:, 1])
    xlim = np.load('../data/VDP/VanDerPol_limitCycle.npy')
    bdry = plt.plot(xlim[0, :], xlim[1, :], color='red', label='ROA boundary')
    plt.show()


def scatterSamples(samples):
    plt.scatter(samples[:, 0], samples[:,1])
    xlim = np.load('../data/VDP/VanDerPol_limitCycle.npy')
    bdry = plt.plot(xlim[0, :], xlim[1, :], color='red', label='ROA boundary')
    plt.show()