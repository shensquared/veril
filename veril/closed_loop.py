import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from pydrake.all import (MathematicalProgram, Polynomial, Expression,
                         MonomialBasis, Jacobian, Evaluate)

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
    if remove_one:
        return np.array([i.ToExpression() for i in MonomialBasis(x, deg)[:-1]])
    else:
        return np.array([i.ToExpression() for i in MonomialBasis(x, deg)])


class ClosedLoopSys(object):

    def __init__(self):
        pass

    def polynomial_dynamics(self, sample_states=None):
        if sample_states is None:
            x = self.sym_x
        else:
            x = sample_states.T
        return self.fx(None, x)

    def init_x_f(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_f = self.polynomial_dynamics()

    def set_syms(self, deg, remove_one=True):
        self.degFeatures = deg
        self.degV = 2 * deg
        self.sym_phi = get_monomials(self.sym_x, deg, remove_one=remove_one)
        sym_dphidx = Jacobian(self.sym_phi, self.sym_x)
        self.sym_eta = sym_dphidx@self.sym_f
        self.remove_one = remove_one

    def features_at_x(self, x):  # x: (num_samples, sys_dim)
        n_samples = x.shape[0]
        phi, eta = [], []
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            eta.append([j.Evaluate(env) for j in self.sym_eta])
        return [np.array(phi), np.array(eta)]

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
        remove_one = self.remove_one
        self.sym_V = V
        self.sym_Vdot = self.set_Vdot(V)
        self.degVdot = self.degV - 1 + self.degf
        deg = int(np.ceil((self.degVdot - self.degV) / 2))
        self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
        psi_deg = int(((2 * deg + self.degV) / 2))
        self.sym_psi = get_monomials(
            self.sym_x, psi_deg, remove_one=remove_one)

    def get_sample_variety_features(self, samples):
        # samples: (num_samples, sys_dim)
        n_samples = samples.shape[0]
        xxd, psi = [], []
        for i in range(n_samples):
            env = dict(zip(self.sym_x, samples[i, :]))
            xxd.append(self.sym_xxd.Evaluate(env))
            psi.append([j.Evaluate(env) for j in self.sym_psi])
        return [np.array(xxd), np.array(psi)]

    def get_v_values(self, samples, V=None):
        if V is None:
            V = self.sym_V
        return np.array([V.Evaluate(dict(zip(self.sym_x, i))) for i in samples])

    def linearized_quadractic_V(self):
        x = self.sym_x
        f = self.sym_f
        env = dict(zip(x, np.zeros(x.shape)))
        A = np.array([[i.Evaluate(env) for i in j]for j in Jacobian(f, x)])
        print('A  %s' % A)
        print('eig of the linearized A matrix %s' % (eig(A)[0]))
        P = solve_lyapunov(A.T, -np.eye(x.shape[0]))
        # print('P %s' % P)
        print('eig of P %s' % (eig(P)[0]))
        V = self.sym_x.T@P@self.sym_x
        return A, P, V

    def P_to_V(self, P, samples=None):
        V0 = self.sym_phi.T@P@self.sym_phi
        Vdot0 = self.set_Vdot(V0)
        H = Jacobian(Vdot0.Jacobian(self.sym_x).T, self.sym_x)
        env = dict(zip(self.sym_x, np.zeros(self.num_states)))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        print('eig of Hessian of Vdot0 %s' % (eig(H)[0]))
        # assert (np.all(eig(H)[0] <= 0))
        if samples is not None:
            V_evals = self.get_v_values(samples, V=V0)
            m = np.percentile(V_evals, 75)
            V = V0 / m
        else:
            V = V0
        Vdot = self.set_Vdot(V)
        return V, Vdot

    def set_Vdot(self, V):
        return V.Jacobian(self.sym_x)@self.sym_f

    def sim_stable_samples(self, x=None, **kwargs):
        event = self.event

        if x is None:
            if 'd' in kwargs:
                d = kwargs['d']
            if 'num_grid' in kwargs:
                num_grid = kwargs['num_grid']
            x = self.get_x(d=d, num_grid=num_grid, slice_idx=slice_idx)

        stableSamples = []
        for i in x:
            # start = time.time()
            sol = integrate.solve_ivp(self.fx, [0, self.int_horizon], i,
                                      events=event)
            # if sol.status == 1:
            # print('event stopping')
            if sol.status != -1 and self.is_at_fixed_pt(sol.y[:, -1]):
                stableSamples.append(i)
            # end = time.time()
            # print(end-start)
        stableSamples = np.array(stableSamples)
        if 'slice_idx' in kwargs:
            return stableSamples[:, kwargs['slice_idx']]
        else:
            return stableSamples

    def sim_stable_samples_for_all_slices(self, d, num_grid):
        for i in self.all_slices:
            samples = self.sim_stable_samples(
                d=d, num_grid=num_grid, slice_idx=i)
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

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-d, d, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == 0, axis=1)]

    def is_at_fixed_pt(self, x):
        # return np.linalg.norm(x) <= self.at_fixed_pt_tol
        return np.allclose(x,0, atol=self.at_fixed_pt_tol)

    def event(self, t, x):
        norm = np.linalg.norm(x)
        out_ = norm - self.int_stop_ub
        in_ = norm - self.int_stop_lb
        return out_ or in_
    event.terminal = True


class VanderPol(ClosedLoopSys):

    def __init__(self):
        self.name = 'VanderPol'
        self.num_states = 2
        self.slice = [0, 1]
        self.all_slices = [[0, 1]]
        self.trueROA = True
        self.degf = 3
        self.init_x_f()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 5
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 20

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


class Pendubot(ClosedLoopSys):

    def __init__(self):
        self.name = 'Pendubot'
        self.num_states = 4
        self.slice = [0, 2]
        self.all_slices = list(
            itertools.combinations(range(self.num_states), 2))
        self.degf = 3
        self.init_x_f()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 15
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 20


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

    def fx(self, t, y):
        [x1, x2, x3, x4] = y
        return np.array([1 * x2,
                         782 * x1 + 135 * x2 + 689 * x3 + 90 * x4,
                         1 * x4,
                         279 * x1 * x3**2 - 1425 * x1 - 257 * x2 + 273 * x3**3
                         - 1249 * x3 - 171 * x4])
