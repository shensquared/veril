import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import samples
from mpl_toolkits.mplot3d import Axes3D


def plot_funnel(V, sys_name, slice_idx, add_title=''):
    file_dir = '../data/' + sys_name
    x = samples.levelsetData(V, slice_idx)[0]
    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='Known ROA Boundary')
    else:
        stable_samples = np.load(file_dir + '/stableSamplesSlice.npy')
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red',
                    label='Simulated Stable Samples')
    plt.plot(x[:, 0], x[:, 1], label='Verified ROA Boundary')
    xlab = 'X' + str(slice_idx[0] + 1)
    ylab = 'X' + str(slice_idx[1] + 1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    leg = plt.legend()
    plt.title(sys_name + add_title)
    plt.show()


def scatterSamples(samples, sys_name, slice_idx, add_title=''):
    file_dir = '../data/' + sys_name
    stable_samples = np.load(file_dir + '/stableSamplesSlice' +
                             str(slice_idx[0] + 1) + str(slice_idx[1] + 1) +
                             '.npy')

    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='yellow', label='ROA boundary')

    plt.scatter(stable_samples[:, 0], stable_samples[
        :, 1], color='red', label='Simulated Stable Samples')
    plt.scatter(samples[:, slice_idx[0]], samples[:, slice_idx[1]],
                label='Samples')
    xlab = 'X' + str(slice_idx[0] + 1)
    ylab = 'X' + str(slice_idx[1] + 1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.title(sys_name + ' ' + add_title)
    plt.show()


def plot3d(V, sys_name, slice_idx, r_max=2, add_title=''):
    thetas = np.linspace(-np.pi, np.pi, 100)
    sym_x = list(V.GetVariables())
    n = thetas.shape[0]
    raddi = np.linspace(.001, r_max, 100)
    CS = np.vstack((np.cos(thetas), np.sin(thetas)))
    x = 0
    y = 0
    z0 = samples.evaluate(np.zeros((1,)), np.vstack((0, 0)), V, sym_x,
                          slice_idx)
    z = z0
    for i in raddi:
        r = i * np.ones(thetas.shape)
        z = np.append(z, samples.evaluate(r, CS, V, sym_x, slice_idx).ravel())
        x = np.append(x, (r * CS[0]).ravel())
        y = np.append(y, (r * CS[1]).ravel())

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x[1:], y[1:], z[1:]
    ax.plot_trisurf(X, Y, Z, linewidth=0.2,
                    cmap=plt.cm.Spectral, antialiased=True)
    xlab = 'X' + str(slice_idx[0] + 1)
    ylab = 'X' + str(slice_idx[1] + 1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    # ax.set_ylim(-r_max, r_max)
    ax.set_zlabel('V')

    levels = np.linspace(1, np.max(np.abs(Z)), 5)
    for i in levels:
        xx = samples.levelsetData(V / i, slice_idx)[0]
        ax.plot(xx[:, 0], xx[:, 1], zs=0, zdir='z', label='levels')

    plt.show()
