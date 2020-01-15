import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from pydrake.all import (MathematicalProgram, Polynomial, Expression,
                         MonomialBasis, Jacobian, Evaluate)

import os
import itertools
import six
import numpy as np
from veril import closed_loop


def get(plant_name):
    if isinstance(plant_name, six.string_types):
        identifier = str(plant_name)
        return globals()[identifier]()


def get_monomials(x, deg, remove_one=False):
    if remove_one:
        return np.array([i.ToExpression() for i in MonomialBasis(x, deg)[:-1]])
    else:
        return np.array([i.ToExpression() for i in MonomialBasis(x, deg)])


class S4CV_Plants(closed_loop.ClosedLoopSys):

    def __init__():
        pass

    def init_x_g_B(self):
        prog = MathematicalProgram()
        # sym_x is defined in the error frame, i.e. sym_x = xbar=x-x0 if using
        # standard notation
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_g = self.dynamic_without_control()
        self.ctrl_B = self.hx()

    def dynamic_without_control(self, sample_states=None):
        if sample_states is None:
            x = self.sym_x
            gx = self.gx(x)
        else:
            x = sample_states.T
            gx = self.gx(x).T
        return gx

    def set_syms(self, degFeatures, degU, remove_one=True):
        self.remove_one = remove_one
        self.degFeatures = degFeatures
        self.degV = 2 * degFeatures
        self.degU = degU
        self.sym_phi = get_monomials(
            self.sym_x, degFeatures, remove_one=remove_one)
        self.sym_dphidx = Jacobian(self.sym_phi, self.sym_x)
        self.sym_ubasis = get_monomials(self.sym_x, degU,
                                        remove_one=remove_one)

    def features_at_x(self, x=None):  # x: (num_samples, sys_dim)
        if x is None:
            x = self.get_x()
        g = self.dynamic_without_control(sample_states=x)
        n_samples = x.shape[0]
        phi, dphidx, ubasis = [], [], []
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i]))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            dphidx.append([[j.Evaluate(env) for j in k]for k in
                           self.sym_dphidx])
            ubasis.append([j.Evaluate(env) for j in self.sym_ubasis])
        features = [g, np.array(phi), np.array(dphidx), np.array(ubasis)]
        model_dir = '../data/' + self.name + '/features_degV'
        file_path = model_dir + str(self.degV) + 'degU' + str(self.degU) + \
            '.npz'
        np.savez_compressed(file_path, g=features[0], phi=features[1],
                            dphidx=features[2], ubasis=features[3])
        return features

    def fx(self, t, y):
        _ = list(itertools.combinations_with_replacement(np.append(1, y),
                                                         self.degU))
        basis = [np.prod(j) for j in _]
        if self.remove_one:
            basis = basis[1:]
        numerical_u_basis = np.array(basis[::-1])
        # print(numerical_u_basis.shape)
        u = (numerical_u_basis@self.u_weights).T
        # theta = y@np.array([1,0])
        # u = (9.81* np.sin(theta + np.pi)).reshape((1,))
        env = dict(zip(self.sym_x, y))
        sym_sol = np.array([j.Evaluate(env) for j in self.u])
        # print(u - sym_sol)
        num_sol = self.gx(y) + self.ctrl_B@u
        return num_sol

    def close_the_loop(self, u_weights):
        self.loop_closed = True
        self.u_weights = u_weights
        self.u = (self.sym_ubasis@u_weights).T
        self.sym_f = self.sym_g + self.ctrl_B@self.u


class PendulumTrig(S4CV_Plants):

    def __init__(self):
        self.name = 'PendulumTrig'
        self.loop_closed = False
        self.num_states = 2
        self.num_inputs = 1
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81

        self.slice = [0, 1]
        self.all_slices = [[0, 1]]
        self.degf = 1

        self.x0 = np.array([np.pi, 0])  # theta=pi, thetadot=0
        self.x0dot = np.zeros((self.num_states,))
        self.init_x_g_B()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 1e10
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 10
        self.d = 2
        self.num_grid = 100

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == 0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [x1, x2] = x
        # thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        # put the origin at the top right
        thetaddot = -self.b * x2 / self.I - \
            self.g * np.sin(x1 + np.pi) / self.l
        return np.array([1 * x2, thetaddot])

    def hx(self, x=None):
        return np.array([[0], [1 / self.I]])

    def is_at_fixed_pt(self, x):
        vel_close = np.isclose(x[-1], 0, atol=self.at_fixed_pt_tol)
        if not vel_close:
            return False
        else:
            y = np.arctan2(np.sin(x[0]), np.cos(x[0]))
            return np.isclose(y, 0, atol=self.at_fixed_pt_tol)

# plant = get('PendulumTrig')
# plant.set_syms(3,2)
# plant.features_at_x()
