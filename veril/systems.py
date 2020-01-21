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
        if 'slice_idx' in kwargs:
            return stableSamples[:, kwargs['slice_idx']]
        else:
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
        # return np.linalg.norm(x) <= self.at_fixed_pt_tol
        return np.allclose(x, self.x0, atol=self.at_fixed_pt_tol)

    def event(self, t, x):
        norm = np.linalg.norm(x)
        in_ = norm - self.int_stop_lb
        if self.int_stop_ub is None:
            out_ = False
        else:
            out_ = norm - self.int_stop_ub
        return out_ or in_
    event.terminal = False


class S4CV_Plants(ClosedLoopSys):

    def __init__():
        pass

    def init_x_g_B(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_g = self.dynamic_without_control()

    def dynamic_without_control(self, sample_states=None):
        if sample_states is None:
            x = self.sym_x
            gx = self.gx(x)
        else:
            x = sample_states.T
            gx = self.gx(x).T
        return gx

    def features_at_x(self, x, file_path):  # x: (num_samples, sys_dim)
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
        if file_path is not None:
            np.savez_compressed(file_path, g=features[0], phi=features[1],
                                dphidx=features[2], ubasis=features[3])
        return features

    def fx(self, t, y):
        u_basis = get_monomials(y - self.x0, self.deg_u, True)
        u = (u_basis@self.u_weights).T
        # u = (9.81* np.sin(y[0] + np.pi)).reshape((1,))
        # u = np.zeros((self.num_inputs))
        num_sol = self.gx(y) + self.hx()@u
        # print(num_sol-[i.Evaluate(dict(zip(self.sym_x,y))) for i in
        #    self.sym_f])
        # print(num_sol)
        return num_sol

    def close_the_loop(self, u_weights):
        self.loop_closed = True
        self.u_weights = u_weights
        self.u = (self.sym_ubasis@u_weights).T
        self.sym_f = self.sym_g + self.hx()@self.u
        # TODO: fix self.degf


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

        self.x0 = np.array([0, 0])  # theta=pi, thetadot=0
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
        return x[~np.all(x == self.x0, axis=1)]

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


class PendulumRecast(S4CV_Plants):

    def __init__(self):
        self.name = 'PendulumRecast'
        self.loop_closed = False
        self.num_states = 3
        self.num_inputs = 1
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81

        self.slice = [1, 2]
        self.all_slices = itertools.combinations(range(self.num_states), 2)
        self.original_coordinate()

        # self.x0 = np.array([np.sin(self.xo0[0]), np.cos(self.xo0[0]),
        #                     self.xo0[1]])
        self.x0 = np.array([0, -1, 0])
        self.x0dot = np.zeros((self.num_states,))
        self.init_x_g_B()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 1e10
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 10
        self.d = 2
        self.num_grid = 100

    def original_coordinate(self):
        prog = MathematicalProgram()
        self.xo = prog.NewIndeterminates(2, "t")
        self.xo0 = [np.pi, 0]

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

    def hx(self, x=None):
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


class VirtualDubins(ClosedLoopSys):

    def __init__(self):
        self.name = 'VirtualDubins'
        self.loop_closed = True

        # parameters
        self.ldot = 1
        self.kv = 0
        self.dt = 1e-3

        self.num_states = 6
        self.init_x_f()

        self.slice = [1, 2]
        self.all_slices = itertools.combinations(range(self.num_states), 2)
        self.x0 = np.zeros((self.num_states,))

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 1e10
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 10
        self.d = 2
        self.num_grid = 100
        self.degf = 3

    def init_x_f(self):
        # TODO: unify with parent class
        prog = MathematicalProgram()
        x = prog.NewIndeterminates(3, "x")
        c = prog.NewIndeterminates(3, "c")
        self.sym_x = np.concatenate((x, c))
        self.sym_f = self.polynomial_dynamics()

    def controller_paras(self):
        A = [[0.667296480803582, -1.30071114303233e-15, -0.169922173102605],
             [2.36315072340782e-14, 0.765660350299141, -1.81433740398421e-13],
             [-0.156887495801010, 7.80454115721781e-14, 0.0405628714680193]]

        B = [[-0.00361904621810242, -5.81436260137465e-14, -0.133112254766150],
             [-1.26904136863127e-13, 0.000321665900839647, -9.25777517950906e-14],
             [0.00103648768038458, -2.42699791299319e-13, 0.0380092356423031]]

        C = [[5.30474466434343e-14, -0.487960601890959, -3.90318131599281e-13],
             [-0.534621938183623, -2.93540009334162e-14, 0.102770198642539]]

        D = [[-2.63750218560237e-13, -0.000204505875506722, -1.85654985333980e-13],
             [-0.00720909658962582, -1.36736538875163e-15, -0.259016984857200]]
        A, B, C, D = np.array(A), np.array(B), np.array(C), np.array(D)
        # TODO: generalize the hard-coded full-state feedback
        return [A, B, C, D]

    def hx(self, x):
        [x1, x2, x3] = x
        return np.array([[1], [x3], [-x2]])

    def fx(self, t, y):
        # the control law is:
        # U = [V ; kb] = [ldot; kv] + uTilde,
        # Where:
        # c(k+1) = A*c(k) + B*x(k)
        # uTilde(k) = C * c(k) + D*x(k)
        # with x = [psi_E ; X_E; Y_E];
        ldot = self.ldot
        kv = self.kv
        [A, B, C, D] = self.controller_paras()

        [x1, x2, x3, c1, c2, c3] = y
        x = [x1, x2, x3]
        c = [c1, c2, c3]

        c_plus = A@c + B@x
        uTilde = C@c + D@x
        U = np.array([ldot, kv]) + uTilde
        [V, kb] = U

        x1dot = V * kb - kv * ldot
        x2dot = V * kb * x3 + V - ldot * np.cos(1 * x1)
        x3dot = -V * kb * x2 + ldot * np.sin(1 * x1)
        cdot = (c_plus - c) / self.dt
        return np.concatenate([[x1dot], [x2dot], [x3dot], cdot])


class VanderPol(ClosedLoopSys):

    def __init__(self):
        self.name = 'VanderPol'
        self.loop_closed = True
        self.num_states = 2
        self.slice = [0, 1]
        self.all_slices = [[0, 1]]
        self.trueROA = True
        self.degf = 3
        self.x0 = np.zeros((self.num_states,))
        self.init_x_f()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 5
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 20
        self.d = 2
        self.num_grid = 100

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
        self.loop_closed = True
        self.num_states = 4
        self.slice = [0, 2]
        self.all_slices = list(
            itertools.combinations(range(self.num_states), 2))
        self.degf = 3
        self.x0 = np.zeros((self.num_states,))
        self.init_x_f()

        self.at_fixed_pt_tol = 1e-3
        self.int_stop_ub = 15
        self.int_stop_lb = self.at_fixed_pt_tol
        self.int_horizon = 20

        self.d = 2
        self.num_grid = 100

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
    return np.array(basis[::-1])

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
