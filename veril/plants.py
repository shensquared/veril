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


def plant_to_closedloop(plant, u_weights):
    system = closed_loop.ClosedLoopSys()
    system.name = plant.name
    system.num_states = plant.num_states
    system.slice = plant.slice
    system.all_slices = plant.all_slices

    system.remove_one = plant.remove_one
    system.degFeatures = plant.degFeatures
    system.degV = plant.degV
    system.sym_phi = plant.sym_phi

    u = (plant.sym_ubasis@u_weights).T
    fx = plant.sym_g + plant.ctrl_B@u
    system.sym_x = plant.sym_x
    system.sym_f = fx
    return system


class S4CV_Plants(object):

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
            x = self.sym_x + self.x0
            gx = self.gx(x)
        else:
            x = sample_states.T
            gx = self.gx(x).T
        return gx - self.x0dot

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
        model_dir = '../data/' + self.name + '/V_u_features'
        file_path = model_dir + '_degV' + \
            str(self.degFeatures) + 'degU' + str(self.degU) + '.npz'
        np.savez_compressed(file_path, g=features[0], phi=features[1],
                            dphidx=features[2], ubasis=features[3])
        return features


class PendulumTrig(S4CV_Plants):

    def __init__(self):
        self.name = 'PendulumTrig'
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

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        x = x + self.x0
        return x[~np.all(x == 0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [x1, x2] = x
        thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        return np.array([1 * x2, thetaddot])

    def hx(self, x=None):
        return np.array([[0], [1 / self.I]])

# plant = get('PendulumTrig')
# plant.set_syms(3,2)
# plant.features_at_x()
