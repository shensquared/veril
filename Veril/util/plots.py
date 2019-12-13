import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import samples
from mpl_toolkits.mplot3d import Axes3D


def plot_funnel(V, sys_name, slice_idx=None):
    file_dir = '../data/' + sys_name
    x = samples.levelsetData(V, slice_idx)[0]
    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='ROAboundary')
    else:
        stable_samples = np.load(file_dir + '/stableSamplesSlice.npy')
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red')
    plt.fill(x[:, 0], x[:, 1])
    plt.show()


def scatterSamples(samples, sys_name, slice_idx=None):
    if slice_idx is None:
        plt.scatter(samples[:, 0], samples[:, 1])
    else:
        plt.scatter(samples[:, slice_idx[0]], samples[:, slice_idx[1]])
    if sys_name is 'VanderPol':
        xlim = np.load('../data/VanderPol/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='ROA boundary')
    plt.show()


def plot3d(V, sys_name, slice_idx=None):
    # n_radii = 8
    n_angles = 100

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
    radii = np.linspace(0.01, 2.0, n_angles)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
    # angles = np.repeat(angles[..., np.newaxis], n_angles, axis=1)

    x = list(V.GetVariables())
    CS = np.vstack((np.cos(angles), np.sin(angles)))


# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.

# Compute z to make the pringle surface.
    z = samples.evaluate(radii, CS, V, x, slice_idx)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # x = np.append(0, (radii*np.cos(angles)).flatten())
    # y = np.append(0, (radii*np.sin(angles)).flatten())
    x=(radii*np.cos(angles)).flatten()
    y=(radii*np.sin(angles)).flatten()

    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)

    plt.show()
