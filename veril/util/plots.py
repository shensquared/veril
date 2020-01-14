import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import samples
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_funnel(V, sys_name, slice_idx, add_title=''):
    id1 = str(slice_idx[0] + 1)
    id2 = str(slice_idx[1] + 1)
    file_dir = '../data/' + sys_name
    x = samples.levelsetData(V, slice_idx)[0]
    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0, :], xlim[1, :],
                        color='red', label='Known ROA Boundary')
    else:
        stable_samples = np.load(file_dir + '/stableSamplesSlice' + id1 + id2 +
                                 '.npy')
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red',
                    label='Simulated Stable Samples')
    plt.plot(x[:, 0], x[:, 1], label='Verified ROA Boundary')
    xlab, ylab = 'X' + id1, 'X' + id2
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    leg = plt.legend()
    plt.title(sys_name + add_title)
    plt.show()


def scatterSamples(samples, sys_name, slice_idx, add_title=''):
    id1 = str(slice_idx[0] + 1)
    id2 = str(slice_idx[1] + 1)

    file_dir = '../data/' + sys_name
    stableSamples_path = file_dir + '/stableSamplesSlice' + id1 + id2 + '.npy'
    if os.path.exists(stableSamples_path):
        stable_samples = np.load(stableSamples_path)
        plt.scatter(stable_samples[:, 0], stable_samples[
            :, 1], color='red', label='Simulated Stable Samples')

    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0], xlim[1], color='yellow', label='ROA boundary')

    plt.scatter(samples[:, slice_idx[0]], samples[:, slice_idx[1]],
                label='Samples')
    xlab, ylab = 'X' + id1, 'X' + id2
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.title(sys_name + ' ' + add_title)
    plt.show()


def plot3d(V, slice_idx, r_max=2, level_sets=False):
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

    ax.set_xlabel('X' + str(slice_idx[0] + 1))
    ax.set_ylabel('X' + str(slice_idx[1] + 1))
    # ax.set_ylim(-r_max, r_max)
    ax.set_zlabel('V')
    if level_sets:
        levels = np.linspace(1, np.max(np.abs(Z)), 5)
        for i in levels:
            xx = samples.levelsetData(V / i, slice_idx)[0]
            ax.plot(xx[:, 0], xx[:, 1], zs=0, zdir='z', label='levels')
    plt.show()


# def phase_portrait(self, ax, ax_max):
#     num = 60
#     u = np.linspace(-ax_max, ax_max, num=num)
#     v = np.linspace(-ax_max, ax_max, num=num)
#     u, v = np.meshgrid(u, v)
#     u, v = u.ravel(), v.ravel()
#     x = -v
#     y = u - v + v * u**2
#     # angles='xy', width=1e-3,scale_units='xy', scale=12, color='r'
#     ax.quiver(u, v, x, y, color='r', width=1e-3, scale_units='x')

# def plot_sim_traj(self, timesteps, full_states, stable_sample,
#                   num_samples=1, scale_time=1):
#     fig, ax = plt.subplots()
#     for i in range(num_samples):
#         sim_traj = self.sim_traj(timesteps, full_states,
#                                  stable_sample, scale_time=scale_time)
#         if full_states:
#             self._phase_portrait(ax, 3)
#             ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
#             ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
#             # ax.axis([-3, 3, -3, 3])
#             plt.xlabel('$x_1$')
#             plt.ylabel('$x_2$', rotation=0)
#             plt.xticks(fontsize=8)
#             plt.yticks(fontsize=8)
#         else:
#             t = np.arange(timesteps)
#             ax.scatter(t, sim_traj, c='b')
#             plt.xlabel('$t$')
#             plt.ylabel('$x_2$', rotation=0)
#             plt.xticks(fontsize=8)
#             plt.yticks(fontsize=8)
#     plt.show()
