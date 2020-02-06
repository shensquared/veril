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


class ClosedLoopSys(object):

    def __init__(self, name, num_states, idx=(0, 1)):
        self.loop_closed = True

        self.name = name
        self.num_states = num_states
        self.slice = idx
        self.x0 = np.zeros((num_states,))
        self.x0dot = np.zeros((num_states,))
        self.all_slices = list(itertools.combinations(range(num_states), 2))
        if hasattr(self, 'special_fixed_pt'):
            self.special_fixed_pt()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 1e5
        self.int_horizon = 10
        self.d = 2
        self.num_grid = 100

    def init_x_f(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_f = self.fx(None, self.sym_x)

    def set_syms(self, deg_ftrs, deg_u, rm_one):
        self.rm_one = rm_one
        self.deg_ftrs = deg_ftrs
        self.degV = 2 * deg_ftrs
        x = self.sym_x
        xbar = self.sym_x - self.x0
        self.sym_phi = get_monomials(xbar, deg_ftrs, rm_one)
        sym_dphidx = Jacobian(self.sym_phi, x)

        if self.loop_closed:
            self.sym_eta = sym_dphidx@self.sym_f
        else:
            self.deg_u = deg_u
            self.sym_dphidx = sym_dphidx
            self.sym_ubasis = get_monomials(xbar, deg_u, True)

    def features_at_x(self, x, file_path):  # x: (num_samples, sys_dim)
        n_samples = x.shape[0]
        phi, eta = [], []
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            eta.append([j.Evaluate(env) for j in self.sym_eta])
        features = [np.array(phi), np.array(eta)]
        if file_path is not None:
            np.savez_compressed(file_path, phi=features[0], eta=features[1])
        return features

    def set_sample_variety_features(self, V):
        # this requires far lower degreed multiplier xxd and consequentially
        # lower degree psi, re-write both
        rm_one = self.rm_one
        x = self.sym_x
        xbar = self.sym_x - self.x0
        self.sym_V = V
        self.sym_Vdot = self.set_Vdot(V)
        self.degVdot = self.degV - 1 + self.degf
        deg = int(np.ceil((self.degVdot - self.degV) / 2))
        self.sym_xxd = (xbar.T@xbar)**(deg)
        psi_deg = int(((2 * deg + self.degV) / 2))
        self.sym_psi = get_monomials(xbar, psi_deg, rm_one)

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
        env = dict(zip(x, self.x0))
        A = np.array([[i.Evaluate(env) for i in j]for j in Jacobian(f, x)])
        print('A  %s' % A)
        print('eig of the linearized A matrix %s' % (eig(A)[0]))
        P = solve_lyapunov(A.T, -np.eye(x.shape[0]))
        print('eig of P %s' % (eig(P)[0]))
        # assert (np.all(eig(P)[0] >= 0))
        V = (self.sym_x - self.x0).T@P@(self.sym_x - self.x0)
        return A, P, V

    def P_to_V(self, P, samples=None):
        V0 = self.sym_phi.T@P@self.sym_phi
        Vdot0 = self.set_Vdot(V0)
        self.VdotHessian(Vdot0)
        if samples is not None:
            V_evals = self.get_v_values(samples, V=V0)
            m = np.percentile(V_evals, 75)
            V = V0 / m
        else:
            V = V0
        Vdot = self.set_Vdot(V)
        return V, Vdot

    def VdotHessian(self, Vdot):
        H = Jacobian(Vdot.Jacobian(self.sym_x).T, self.sym_x)
        env = dict(zip(self.sym_x, self.x0))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        print('eig of Hessian of Vdot0 %s' % (eig(H)[0]))
        # assert (np.all(eig(H)[0] <= 0))

    def set_Vdot(self, V):
        return V.Jacobian(self.sym_x)@self.sym_f

    def forward_sim(self, x, **kwargs):
        event = kwargs['event'] if 'event' in kwargs else self.event
        int_horizon = kwargs['int_horizon'] if 'int_horizon' in kwargs else \
            self.int_horizon
        sol = integrate.solve_ivp(self.fx, [0, int_horizon], x, events=event)
        return sol

    def random_sample(self, n):
        return np.random.randn(n, self.num_states)

    def sample_stable_inits(self, x=None, **kwargs):
        event = self.event
        if x is None:
            x = self.get_x(**kwargs)
        stableSamples = []
        for i in x:
            # start = time.time()
            sol = self.forward_sim(i, **kwargs)
            # if sol.status == 1:
            # print('event stopping')
            # if sol.status != 1:
            #     print('init %s' % i)
            #     print('final %s' % sol.y[:, -1])
            if sol.status != -1 and self.is_at_fixed_pt(sol.y[:, -1]):
                stableSamples.append(i)
            # end = time.time()
            # print(end-start)
        stableSamples = np.array(stableSamples)
        name = '../data/' + self.name + '/stableSamples'
        if 'slice_idx' in kwargs:
            stableSamples = stableSamples[:, kwargs['slice_idx']]
            name = name + 'Slice' + str(i[0] + 1) + str(i[1] + 1) + '.npy'
        np.save(name, stableSamples)
        return stableSamples

    def sample_stable_inits_for_all_slices(self, **kwargs):
        for i in self.all_slices:
            samples = self.sample_stable_inits(slice_idx=i, **kwargs)
            name = '../data/' + self.name + '/stableSamplesSlice' + \
                str(i[0] + 1) + str(i[1] + 1) + '.npy'
            np.save(name, samples)

    def get_x(self, **kwargs):
        d = kwargs['d'] if 'd' in kwargs else self.d
        num_grid = kwargs[
            'num_grid'] if 'num_grid' in kwargs else self.num_grid

        x0 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x0, x0)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == self.x0, axis=1)]

    def is_at_fixed_pt(self, x):
        return np.allclose(x, self.x0, atol=self.at_fixed_pt_tol)

    def event(self, t, x):
        norm = np.linalg.norm(x)
        # in_ = x - self.x0
        if self.int_stop_ub is not None:
            out_ = norm - self.int_stop_ub
        return out_
    event.terminal = True


class S4CV_Plants(ClosedLoopSys):

    def __init__(self, name, num_states, num_inputs, idx=(0, 1)):
        super().__init__(name, num_states, idx=idx)
        self.num_inputs = num_inputs
        self.loop_closed = False

    def init_x_g_B(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_g = self.gx(self.sym_x)

    def features_at_x(self, x, file_path):  # x: (num_samples, sys_dim)
        if x is None:
            x = self.get_x()
        g = self.gx(x.T).T  # (just so 1st dim is # of samples)
        n_samples = x.shape[0]
        B, phi, dphidx, ubasis = [], [], [], []
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i]))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            dphidx.append([[j.Evaluate(env) for j in k]for k in
                           self.sym_dphidx])
            ubasis.append([j.Evaluate(env) for j in self.sym_ubasis])
            if hasattr(self, 'B_noneConstant'):
                B.append(self.hx(x[i]))
        features = [g, np.array(phi), np.array(dphidx), np.array(ubasis)]
        if file_path is not None:
            if hasattr(self, 'B_noneConstant'):
                np.savez_compressed(file_path, g=features[0], B=np.array(B),
                                    phi=features[1], dphidx=features[2],
                                    ubasis=features[3])
                features = [g, np.array(B), np.array(phi), np.array(dphidx),
                np.array(ubasis)]
            else:
                np.savez_compressed(file_path, g=features[0], phi=features[1],
                                    dphidx=features[2], ubasis=features[3])
        return features

    def fx(self, t, y):
        u_basis = get_monomials(y - self.x0, self.deg_u, True)
        u = (u_basis@self.u_weights).T
        # u = np.zeros((self.num_inputs))
        num_sol = self.gx(y) + self.hx(y)@u
        # print(num_sol-[i.Evaluate(dict(zip(self.sym_x,y))) for i in
        #    self.sym_f])
        # print(num_sol)
        return num_sol

    def close_the_loop(self, u_weights):
        self.loop_closed = True
        self.u_weights = u_weights
        self.u = (self.sym_ubasis@u_weights).T
        self.sym_f = self.sym_g + self.hx(self.sym_x)@self.u
        # TODO: fix self.degf


class PendulumTrig(S4CV_Plants):

    def __init__(self):
        super().__init__('PendulumTrig', 2, 1)
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81
        self.init_x_g_B()  # theta=pi, thetadot=0

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == self.x0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [x1, x2] = x
        # thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        # put the origin at the top right
        x2dot = -self.b * x2 / self.I - self.g * np.sin(x1 + np.pi) / self.l
        return np.array([1 * x2, x2dot])

    def hx(self, x):
        return np.array([[0], [1 / self.I]])

    def is_at_fixed_pt(self, x):
        vel_close = np.isclose(x[-1], 0, atol=self.at_fixed_pt_tol)
        if not vel_close:
            return False
        else:
            y = np.arctan2(np.sin(x[0]), np.cos(x[0]))
            return np.isclose(y, 0, atol=self.at_fixed_pt_tol)

    # for debugging only, comment out
    # def fx(self, t, y):
        # u = (9.81* np.sin(y[0] + np.pi)).reshape((1,))
        # u=np.zeros((1,))
        # num_sol = self.gx(y) + self.hx(y)@u
        # return num_sol


class PendulumRecast(S4CV_Plants):

    def __init__(self):
        super().__init__('PendulumRecast', 3, 1, idx=(1, 2))
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81
        self.init_x_g_B()

    def special_fixed_pt(self):
        prog = MathematicalProgram()
        self.xo = prog.NewIndeterminates(2, "t")
        self.xo0 = [np.pi, 0]
        self.x0 = np.array([0, -1, 0])

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([np.sin(x1), np.cos(x1), x2]).T  # (num_grid**2,3)
        return x[~np.all(x == self.x0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [s, c, thetadot] = x
        # thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        # put the origin at the top right
        sdot = c * thetadot
        cdot = -s * thetadot
        thetaddot = -self.b * thetadot / self.I - self.g * s / self.l
        return np.array([sdot, cdot, thetaddot])

    def hx(self, x):
        return np.array([[0], [0], [1 / self.I]])

    # def is_at_fixed_pt(self, x):
    #     vel_close = np.isclose(x[-1], 0, atol=self.at_fixed_pt_tol)
    #     if not vel_close:
    #         return False
    #     else:
    #         y = np.arctan2(x[0], x[1])
    #         return np.isclose(y, np.pi, atol=self.at_fixed_pt_tol)

    def random_sample(self, n):
        x1, x2 = np.random.randn(n,), np.random.randn(n,)
        x = np.array([np.sin(x1), np.cos(x1), x2]).T  # (n,3)
        return x

    def poly_to_orig(self, func=None):
        t = self.xo
        env = dict(zip(self.sym_x, [np.sin(1 * t[0]), np.cos(1 * t[0]), t[1]]))
        if func is not None:
            func = func.Substitute(env)
            return func, env
        else:
            return env

    def VdotHessian(self, Vdot):
        Vdot, _ = self.poly_to_orig(func=Vdot)
        H = Jacobian(Vdot.Jacobian(self.xo).T, self.xo)
        env = dict(zip(self.xo, self.xo0))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        print('eig of Hessian of Vdot0 %s' % (eig(H)[0]))
        # assert (np.all(eig(H)[0] <= 0))

    def u_scaling_reg(self):
        base = [1., 1., 100.]
        return np.array([get_monomials(base, self.deg_u, True)])

    def debugging_V(self, V):
        up = [np.array([0, -1, 0])]
        down = np.array([np.array([0, 1, 0])])
        print('up V value %s' % system.get_v_values(up, V))
        print('down V value %s' % system.get_v_values(down, V))


class DubinsRecast(S4CV_Plants):

    def __init__(self):
        super().__init__('DubinsRecast', 4, 2, idx=(1, 2))
        # parameters
        self.ldot = 2
        self.kv = 0
        self.init_x_g_B()
        self.B_noneConstant = True
        self.all_slices = list(itertools.combinations(range(3), 2))

    def special_fixed_pt(self):
        prog = MathematicalProgram()
        self.xo = prog.NewIndeterminates(3, "t")
        self.xo0 = np.zeros((3,))
        self.x0 = np.array([0, 1, 0, 0])

    def get_x(self, d=1, num_grid=100, slice_idx=None):
        x1 = np.linspace(-np.pi/2, np.pi/2, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([np.sin(x1), np.cos(x1), x2, x2]).T  # (num_grid**2,4)
        return x[~np.all(x == self.x0, axis=1)]

    def gx(self, x):
        [s, c, xe, ye] = x
        ldot = self.ldot
        kv = self.kv

        thetadot = - kv * ldot
        sdot = c * thetadot
        cdot = -s * thetadot
        xedot = - ldot * c
        yedot = ldot * s
        return np.array([sdot, cdot, xedot, yedot])

    def hx(self, x):
        [s, c, xe, ye] = x
        # if len(x.shape) == 1: #symbolic
            # return np.array([[c, 0], [-s, 0], [ye, 1], [-xe, 0]])
        # else:
            # z = np.zeros(s.shape)
            # one = np.ones(s.shape)
            # return np.array([[c, z], [-s, z], [ye, one], [-xe, z]])
        return np.array([[c, 0], [-s, 0], [ye, 1], [-xe, 0]])

    def random_sample(self, n):
        m = np.pi/2
        theta = np.random.uniform(low=-m, high=m, size=(1, n))
        # x1 = np.random.randn(2, n)
        x1 = np.random.uniform(low=-1, high=1, size=(2, n))
        x = np.vstack((np.sin(theta), np.cos(theta), x1)).T  # (n,4)
        return x

    def poly_to_orig(self, func=None):
        t = self.xo
        env = dict(zip(self.sym_x, [np.sin(1 * t[0]), np.cos(1 * t[0]), t
                                    [1], t[2]]))
        if func is not None:
            func = func.Substitute(env)
            return func, env
        else:
            return env

    def VdotHessian(self, Vdot):
        Vdot, _ = self.poly_to_orig(func=Vdot)
        H = Jacobian(Vdot.Jacobian(self.xo).T, self.xo)
        env = dict(zip(self.xo, self.xo0))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        print('eig of Hessian of Vdot0 %s' % (eig(H)[0]))
        # assert (np.all(eig(H)[0] <= 0))


class VirtualDubins(ClosedLoopSys):

    def __init__(self):
        super().__init__('VirtualDubins', 6, idx=(1, 2))
        # parameters
        self.ldot = 2
        self.kv = 0
        self.init_x_f()

    def init_x_f(self):
        # TODO: unify with parent class
        prog = MathematicalProgram()
        x = prog.NewIndeterminates(3, "x")
        c = prog.NewIndeterminates(3, "c")
        self.sym_x = np.concatenate((x, c))
        self.sym_f = self.fx(None, self.sym_x)

    def controller_paras(self):
        A = [[-5.10497878620518, 1.55202248444586e-08, 10720.8082964261],
             [9.82019782286421e-11, -4.08463276380658, 5.36820301819946e-07],
             [-0.00020225192678231, 3.61384860187543e-11, -24.9338316399476]]

        B = [[10.0727437123573, -3.9325123062356e-11, 4.7460442086702],
             [3.47616559807745e-10, -1.06249843508111, 2.85319375555132e-09],
             [0.21665920969022, 1.43593402212703e-11, 1.06860976714098]]

        C = [[1.08557076610184e-13, 0.0297791141254415, -5.63585841597754e-09],
             [0.00368360435364282, -1.17459038425304e-11, -7.85135434689545]]

        D = [
            [-4.5885769930001e-12, -1.21070877556519e-05, -2.14675996305827e-11],
            [-0.00484908128438563, 2.61735562903613e-15, -0.00364756231283482]]

        A, B, C, D = np.array(A), np.array(B), np.array(C), np.array(D)
        return [A, B, C, D]

    def fx(self, t, y):
        ldot = self.ldot
        kv = self.kv
        [A, B, C, D] = self.controller_paras()

        [x1, x2, x3, c1, c2, c3] = y
        x = [x1, x2, x3]
        c = [c1, c2, c3]

        cdot = A@c + B@x
        uTilde = C@c + D@x
        U = np.array([ldot, kv]) + uTilde
        [V, kb] = U

        x1dot = V * kb - kv * ldot
        x2dot = V * kb * x3 + V - ldot * np.cos(1 * x1)
        x3dot = -V * kb * x2 + ldot * np.sin(1 * x1)

        return np.concatenate([[x1dot], [x2dot], [x3dot], cdot])

    def get_x(self, d=.5, num_grid=100):
        theta = np.linspace(-np.pi, np.pi, num_grid)
        x = np.linspace(-d, d, num_grid)
        x = np.array([i.ravel() for i in np.meshgrid(theta, x, x, x, x, x)]).T
        return x[~np.all(x == self.x0, axis=1)]

    def random_sample(self, n):
        u_init = np.zeros((3, n))
        m = np.pi
        theta = np.random.uniform(low=-m, high=m, size=(1, n))
        # x1 = np.random.randn(2, n)
        x1 = np.random.uniform(low=-1, high=1, size=(2, n))
        x = np.vstack((theta, x1, u_init)).T  # (n,6)
        return x


class VirtualDubins3d(ClosedLoopSys):

    def __init__(self):
        super().__init__('VirtualDubins3d', 3, idx=(1, 2))
        # parameters
        self.ldot = 2
        self.kv = 0
        self.init_x_f()
        self.degf = 4
        self.at_fixed_pt_tol = 5e-2
        self.int_horizon = 100
        self.int_stop_ub = 1e2

    def open_loop(self, x, u):
        ldot = self.ldot
        kv = self.kv
        [x1, x2, x3] = x
        [V, kb] = u
        x1dot = V * kb - kv * ldot
        # x2dot = V * kb * x3 + V - ldot * np.cos(1 * x1)
        # x3dot = -V * kb * x2 + ldot * np.sin(1 * x1)
        s = x1 - x1**3 / 6
        c = 1 - x1**2 / 2
        x2dot = V * kb * x3 + V - ldot * c
        x3dot = -V * kb * x2 + ldot * s
        return np.concatenate([[x1dot], [x2dot], [x3dot]])

    def fx(self, t, y):
        [x1, x2, x3] = y
        # # simpler controller:
        # # U = [V;kb], where
        # # V = ||[xE;yE]|| (or V = sqrt(xE^2 + yE^2) ), and
        # # kb = -yE.
        V = x2**2 + x3**2 + self.ldot
        kb = -x3
        u = np.array([V, kb])
        return self.open_loop(y, u)

    def get_x(self, d=.5, num_grid=200):
        theta = np.linspace(-np.pi / 30, np.pi / 30, num_grid)
        x = np.linspace(-d, d, num_grid)
        x = np.array([i.ravel() for i in np.meshgrid(theta, x, x)]).T
        return x[~np.all(x == self.x0, axis=1)]

    def random_sample(self, n):
        m = np.pi / 20
        theta = np.random.uniform(low=-m, high=m, size=(1, n))
        # x1 = np.random.randn(2, n)
        x1 = np.random.uniform(low=-.5, high=.5, size=(2, n))
        x = np.vstack((theta, x1)).T  # (n,6)
        return x

    # def is_at_fixed_pt(self, x):
    #     vel_close = np.isclose(x[-2:], 0, atol=self.at_fixed_pt_tol)
    #     if not np.all(vel_close):
    #         return False
    #     else:
    #         y = np.arctan2(np.sin(x[0]), np.cos(x[0]))
    #         return np.isclose(y, 0, atol=self.at_fixed_pt_tol)


class VanderPol(ClosedLoopSys):

    def __init__(self):
        super().__init__('VanderPol', 2)
        self.init_x_f()
        self.degf = 3
        self.int_stop_ub = 5
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
        super().__init__('Pendubot', 4, idx=(0, 2))
        self.init_x_f()
        self.degf = 3
        self.int_stop_ub = 15
        self.int_horizon = 20

    def get_x(self, **kwargs):
        d = kwargs['d'] if 'd' in kwargs else self.d
        num_grid = kwargs['num_grid'] if 'num_grid' in kwargs else \
            self.num_grid

        x0 = np.linspace(-d, d, num_grid)
        if 'slice_idx' in kwargs:
            x1, x2 = np.meshgrid(x0, x0)
            x1, x2 = x1.ravel(), x2.ravel()
            x = np.zeros((x1.shape[0], 4))  # (num_grid**4,4)
            x[:, kwargs['slice_idx']] = np.array([x1, x2]).T
        else:
            x1, x2, x3, x4 = np.meshgrid(x0, x0, x0, x0)
            x1, x2, x3, x4 = x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel()
            x = np.array([x1, x2, x3, x4]).T  # (num_grid**4,4)
        return x[~np.all(x == self.x0, axis=1)]

    def fx(self, t, y):
        [x1, x2, x3, x4] = y
        return np.array([1 * x2,
                         782 * x1 + 135 * x2 + 689 * x3 + 90 * x4,
                         1 * x4,
                         279 * x1 * x3**2 - 1425 * x1 - 257 * x2 + 273 * x3**3
                         - 1249 * x3 - 171 * x4])


def get(system_name):
    if isinstance(system_name, six.string_types):
        identifier = str(system_name)
        return globals()[identifier]()


def get_system(sys_name, deg_ftrs, deg_u, rm_one):
    system = get(sys_name)
    system.set_syms(deg_ftrs, deg_u, rm_one)
    return system


def get_monomials(x, deg, rm_one):
    c = 1 if isinstance(x[0], float) else Expression(1)
    _ = itertools.combinations_with_replacement(np.append(c, x), deg)
    basis = [np.prod(j) for j in _]
    if rm_one:
        basis = basis[1:]
    # if rm_one:
    #     print(np.array([i.ToExpression() for i in MonomialBasis(x, deg)[:-1]]))
    # else:
    #     print(np.array([i.ToExpression() for i in MonomialBasis(x, deg)]))
    # print(basis[::-1])
    return np.array(basis[:: -1])

    # def levelset_features(self, V, sigma_deg):
    #     self.sym_V = V
    #     self.sym_Vdot = self.sym_V.Jacobian(self.sym_x) @ self.sym_f
    #     self.degVdot = Polynomial(self.sym_Vdot, self.sym_x).TotalDegree()
    #     deg = int(np.floor((sigma_deg + self.degVdot - self.degV) / 2))
    #     self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
    #     self.sym_sigma = get_monomials(self.sym_x, sigma_deg)
    #     psi_deg = int(np.floor(max(2 * deg + self.degV, sigma_deg +
    #                                self.degVdot) / 2))
    #     self.sym_psi = get_monomials(self.sym_x, psi_deg, rm_one)

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
