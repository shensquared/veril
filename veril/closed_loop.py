import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
from pydrake.all import (MathematicalProgram, Polynomial, Expression,
                         SolutionResult, MonomialBasis, Variables, Solve,
                         Jacobian, Evaluate, Substitute, MosekSolver)

import os
import itertools
import six
# import time
import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
from scipy import integrate

from veril.util.plots import *


def get(system_name):
    if isinstance(system_name, six.string_types):
        identifier = str(system_name)
        return globals()[identifier]()


def get_monomials(x, deg, remove_one=False):
    # y = list(itertools.combinations_with_replacement(np.append(pow(x[0], 0),
    # x),deg))
    # basis = [np.prod(j) for j in y]
    # return np.stack(basis)
    return np.array([i.ToExpression() for i in MonomialBasis(x, deg)])


class ClosedLoopSys(object):

    def __init__():
        pass

    def set_features(self, VFeatureDeg):
        self.degV = 2 * VFeatureDeg
        self.sym_f = self.polynomial_dynamics()
        self.sym_phi = get_monomials(self.sym_x, VFeatureDeg)
        sym_dphidx = Jacobian(self.sym_phi, self.sym_x)
        self.sym_eta = sym_dphidx@self.sym_f

    def train_for_V_features(self, x):  # x: (num_samples, sys_dim)
        # f = self.polynomial_dynamics(sample_states=x)
        n_samples = x.shape[0]
        phi_dim = self.sym_phi.shape[0]
        phi = np.zeros((n_samples, phi_dim))
        # dphidx = np.zeros((n_samples, phi_dim, self.num_states))
        eta = np.zeros((n_samples, phi_dim))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi[i, :] = [i.Evaluate(env) for i in self.sym_phi]
            # dphidx[i, :, :] = [[i.Evaluate(env) for i in j]for j in
            # self.sym_dphidx]
            eta[i, :] = [i.Evaluate(env) for i in self.sym_eta]
        return [phi, eta]

    # def levelset_features(self, V, sigma_deg):
    #     self.sym_V = V
    #     self.sym_Vdot = self.sym_V.Jacobian(self.sym_x) @ self.sym_f
    #     self.degVdot = Polynomial(self.sym_Vdot, self.sym_x).TotalDegree()
    #     deg = int(np.floor((sigma_deg + self.degVdot - self.degV) / 2))
    #     self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
    #     self.sym_sigma = get_monomials(self.sym_x, sigma_deg)
    #     psi_deg = int(np.floor(max(2 * deg + self.degV, sigma_deg +
    #                                self.degVdot) / 2))
    #     self.sym_psi = get_monomials(self.sym_x, psi_deg, remove_one=False)

    # def get_levelset_features(self, x):
    #     # x: (num_samples, sys_dim)
    #     n_samples = x.shape[0]
    #     V = np.zeros((n_samples, 1))
    #     Vdot = np.zeros((n_samples, 1))
    #     xxd = np.zeros((n_samples, 1))
    #     psi = np.zeros((n_samples, self.sym_psi.shape[0]))
    #     sigma = np.zeros((n_samples, self.sym_sigma.shape[0]))
    #     for i in range(n_samples):
    #         env = dict(zip(self.sym_x, x[i, :]))
    #         V[i, :] = self.sym_V.Evaluate(env)
    #         Vdot[i, :] = self.sym_Vdot.Evaluate(env)
    #         xxd[i, :] = self.sym_xxd.Evaluate(env)
    #         psi[i, :] = [i.Evaluate(env) for i in self.sym_psi]
    #         sigma[i, :] = [i.Evaluate(env) for i in self.sym_sigma]
    #     return [V, Vdot, xxd, psi, sigma]

    def set_sample_variety_features(self, V):
        # this requires far lower degreed multiplier xxd and consequentially
        # lower degree psi, re-write both
        self.sym_V = V
        self.sym_Vdot = self.sym_V.Jacobian(self.sym_x) @ self.sym_f
        # self.sym_Vdot = Polynomial(self.sym_Vdot, self.sym_x)
        self.degVdot = self.degV - 1 + self.degf
        deg = int(np.ceil((self.degVdot - self.degV) / 2))
        self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
        psi_deg = int(((2 * deg + self.degV) / 2))
        self.sym_psi = get_monomials(self.sym_x, psi_deg, remove_one=False)

    def get_sample_variety_features(self, samples):
        # samples: (num_samples, sys_dim)
        n_samples = samples.shape[0]
        xxd = np.zeros((n_samples, 1))
        psi = np.zeros((n_samples, self.sym_psi.shape[0]))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, samples[i, :]))
            xxd[i, :] = self.sym_xxd.Evaluate(env)
            psi[i, :] = [i.Evaluate(env) for i in self.sym_psi]
        return [xxd, psi]

    def get_v_values(self, samples, V=None):
        if V is None:
            V = self.sym_V
        return np.array([V.Evaluate(dict(zip(self.sym_x, i))) for i in samples])

    def linearized_A_and_P(self):
        x = self.sym_x
        f = self.polynomial_dynamics()
        J = Jacobian(f, x)
        env = dict(zip(x, np.zeros(x.shape)))

        A = np.array([[i.Evaluate(env) for i in j]for j in J])
        print('A  %s' % A)
        print('eig of the linearized A matrix %s' % (eig(A)[0]))
        P = solve_lyapunov(A.T, -np.eye(x.shape[0]))
        # print('P %s' % P)
        print('eig of P %s' % (eig(P)[0]))
        return A, P

    def rescale_V(self, P, samples):
        V0 = self.sym_phi.T@P@self.sym_phi
        Vdot0 = self.set_Vdot(V0)
        H = Jacobian(Vdot0.Jacobian(self.sym_x).T, self.sym_x)
        env = dict(zip(self.sym_x, np.zeros(self.num_states)))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        print('eig of Hessian of Vdot0 %s' % (eig(H)[0]))
        assert (np.all(eig(H)[0] <= 0))
        V_evals = self.get_v_values(samples, V=V0)
        m = np.percentile(V_evals, 75)
        V = V0 / m
        Vdot = self.set_Vdot(V)
        return V, Vdot

    def set_Vdot(self, V):
        return V.Jacobian(self.sym_x)@self.sym_f

    def sim_stable_samples(self, d, num_grid, slice_idx=None):
        def event(t, x): return np.linalg.norm(x) - 15
    def sim_stable_samples(self, d, num_grid, slice_idx=None, x=None):
    def sim_stable_samples(self, d, num_grid, slice_idx=None, x=None,
                           slice_only=False):
        if slice_idx is not None:
            assert slice_only

        def event(t, x):
            norm = np.linalg.norm(x)
            out_ = norm - 15
            in_ = norm - 1e-8
            return out_ or in_
        event.terminal = True
        if x is None:
            x = self.get_x(d=d, num_grid=num_grid, slice_idx=slice_idx)
        stableSamples = []

        for i in x:
            # start = time.time()
            sol = integrate.solve_ivp(self.fx, [0, 15], i, events=event)
            # if sol.status == 1:
            # print('event stopping')
            if sol.status != -1 and (np.linalg.norm(sol.y[:, -1])) <= 1e-7:
                stableSamples.append(i)
            # end = time.time()
            # print(end-start)
        stableSamples = np.array(stableSamples)
        if slice_only:
            return stableSamples[:, slice_idx]
        else:
            return stableSamples

    def sim_stable_samples_for_all_slices(self, d, num_grid):
        for i in self.all_slices:
            samples = self.sim_stable_samples(d, num_grid, slice_idx=i)
            name = '../data/' + self.name + '/stableSamplesSlice' + \
                str(i[0] + 1) + str(i[1] + 1) + '.npy'
            np.save(name, samples)

    def scatter_stable_samples(self, slice_idx=None):
        if slice_idx is None:
            [scatterSamples(np.zeros((1, self.num_states)), self.name, i)
             for i in self.all_slices]
        else:
            scatterSamples(np.zeros((1, self.num_states)),
                           self.name, slice_idx)


class VanderPol(ClosedLoopSys):

    def __init__(self):
        self.name = 'VanderPol'
        self.num_states = 2
        self.slice = [0, 1]
        self.all_slices = [[0, 1]]
        self.trueROA = True
        self.degf = 3
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-d, d, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == 0, axis=1)]

    def polynomial_dynamics(self, sample_states=None):
        if sample_states is None:
            sym_x = self.sym_x
            return -np.array([sym_x[1], -sym_x[0] - sym_x[1] * (sym_x[0]**2 -
                                                                1)])
        else:
            x = sample_states.T
            f = - np.array([x[1, :], (1 - x[0, :]**2) * x[1, :] - x[0, :]]).T
            return f

    def knownROA(self):
        x = self.sym_x
        x1 = x[0]
        x2 = x[1]
        V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
            (0.18442) * x2**2 + (0.016538) * x2**4 + \
            (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
            (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1
        return V

    def fx(self, t, y):
        return - np.array([y[1], (1 - y[0]**2) * y[1] - y[0]])

    def outward_vdp(self, t, y):
        return - self.fx(t, y)

    def phase_portrait(self, ax, ax_max):
        num = 60
        u = np.linspace(-ax_max, ax_max, num=num)
        v = np.linspace(-ax_max, ax_max, num=num)
        u, v = np.meshgrid(u, v)
        u, v = u.ravel(), v.ravel()
        x = -v
        y = u - v + v * u**2
        # angles='xy', width=1e-3,scale_units='xy', scale=12, color='r'
        ax.quiver(u, v, x, y, color='r', width=1e-3, scale_units='x')

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                self._phase_portrait(ax, 3)
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.scatter(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()


class Pendubot(ClosedLoopSys):

    def __init__(self):
        self.name = 'Pendubot'
        self.num_states = 4
        self.slice = [0, 2]
        self.all_slices = list(
            itertools.combinations(range(self.num_states), 2))
        self.degf = 3
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")

    def get_x(self, d=2, num_grid=100, slice_idx=None):
        x1 = np.linspace(-d, d, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        if slice_idx is None:
            x3 = np.linspace(-d, d, num_grid)
            x4 = np.linspace(-d, d, num_grid)
            x1, x2, x3, x4 = np.meshgrid(x1, x2, x3, x4)
            x1, x2, x3, x4 = x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel()
            x = np.array([x1, x2, x3, x4]).T  # (num_grid**4,4)
        else:
            x1, x2 = np.meshgrid(x1, x2)
            x1, x2 = x1.ravel(), x2.ravel()
            x = np.zeros((x1.shape[0], 4))  # (num_grid**4,4)
            x[:, slice_idx] = np.array([x1, x2]).T
        return x[~np.all(x == 0, axis=1)]

    def polynomial_dynamics(self, sample_states=None):
        if sample_states is None:
            return self.fx(None, self.sym_x)
        else:
            x = sample_states.T
            x1 = x[0, :]
            x2 = x[1, :]
            x3 = x[2, :]
            x4 = x[3, :]
            return self.fx(None, [x1, x2, x3, x4]).T

    def fx(self, t, y):
        [x1, x2, x3, x4] = y
        return np.array([1 * x2, 782 * x1 + 135 * x2 + 689 * x3 + 90 * x4, 1 * x4,
                         279 * x1 * x3**2 - 1425 * x1 - 257 * x2 + 273 *
                         x3**3 - 1249 * x3 - 171 * x4])
