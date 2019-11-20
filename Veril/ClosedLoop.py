import os
import itertools

import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
from pydrake.all import (MathematicalProgram, Polynomial,
                         Expression, SolutionResult, MonomialBasis,
                         Variables, Solve, Jacobian, Evaluate,
                         RealContinuousLyapunovEquation, Substitute,
                         MosekSolver, tanh, pow)

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
from scipy import integrate

from keras import backend as K
from keras.utils import CustomObjectScope
from keras.models import load_model

from Veril import Plants
from Veril.CustomLayers import JanetController
import itertools


def get_NNorCL(NNorCL='CL', **kwargs):
    num_samples = kwargs['num_samples']
    num_units = kwargs['num_units']
    timesteps = kwargs['timesteps']
    dt = kwargs['dt']
    obs_idx = kwargs['obs_idx']
    tag = kwargs['tag']
    plant_name = kwargs['plant_name']

    # dirname = os.path.dirname(__file__)
    dirname = os.path.join('/users/shenshen/Veril/data/')
    model_file_name = dirname + plant_name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + tag

    with CustomObjectScope({'JanetController': JanetController}):
        model = load_model(model_file_name + '.h5')
    print(model.summary())
    if NNorCL is 'NN':
        return model, model_file_name
    elif NNorCL is 'CL':
        for this_layer in model.layers:
            if hasattr(this_layer, 'cell'):
                return [this_layer, model_file_name]


def call_CLsys(CL, tm1, num_samples):
    inputs = K.placeholder()
    states = [K.placeholder(shape=(num_samples, CL.cell.num_plant_states)),
              K.placeholder(shape=(num_samples, CL.cell.units))]
    [x_tm2, c_tm2] = CL.cell.call(inputs, states, training=False)[1]
    feed_dict = dict(zip(states, tm1))
    sess = K.get_session()
    x_tm2 = sess.run(x_tm2, feed_dict=feed_dict)
    c_tm2 = sess.run(c_tm2, feed_dict=feed_dict)
    return [x_tm2, c_tm2]


def batchSim(CL, timesteps, num_samples=10000, init=None):
    """return two sets of initial conditions based on the simulated results.
    One set is the stable trajectory and the other set is the unstable one.

    Args:
        CL (TYPE): Description
        timesteps (TYPE): Description
        init (TYPE): Description
        num_samples (TYPE): Description
    """
    if init is None:
        x = np.random.randn(num_samples, CL.cell.num_plant_states)
        c = np.zeros((num_samples, CL.cell.units))
        init = [x, c]

    for i in range(timesteps):
        init = call_CLsys(CL, init, num_samples)
    return init


def originalSysInitialV(CL):
    plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    A0 = CL.linearize()
    full_dim = plant.num_states + CL.units
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(full_dim, "x")
    if not plant.manifold:
        S0 = solve_lyapunov(A0.T, -np.eye(full_dim))
    else:
        P = prog.NewSymmetricContinuousVariables(full_dim, "P")
        prog.AddPositiveSemidefiniteConstraint(P)
        prog.AddPositiveSemidefiniteConstraint(P + P.T)
        # V = x.T@P@x
        Vdot = x.T@P@A0@x + x.T@A0.T@P@x
        r = prog.NewContinuousVariables(1, "r")[0]
        Vdot = Vdot + r * plant.get_manifold(x)
        slack = prog.NewContinuousVariables(1, "s")[0]
        prog.AddConstraint(slack >= 0)

        prog.AddSosConstraint(-Vdot - slack * x.T@np.eye(full_dim)@x)
        prog.AddCost(-slack)
        solver = MosekSolver()
        solver.set_stream_logging(False, "")
        result = solver.Solve(prog, None, None)
        # print(result.get_solution_result())
        assert result.is_success()
        slack = result.GetSolution(slack)
        print('slack is %s' % (slack))
        S0 = result.GetSolution(P)
    print('eig of orignal A  %s' % (eig(A0)[0]))
    print('eig of orignal SA+A\'S  %s' % (eig(A0.T@S0 + S0@A0)[0]))
    return x.T@S0@x


def GetMonomials(x, deg, remove_one=False):
    # y = list(itertools.combinations_with_replacement(np.append(pow(x[0], 0), x),
    #                                                  deg))
    # basis = [np.prod(j) for j in y]
    # if remove_one:
    #     return np.stack(basis)[1:]
    # else:
    #     return np.stack(basis)

    return np.array([i.ToExpression() for i in MonomialBasis(x, deg)])


class ClosedLoopSys(object):

    def __init__():
        pass

    def set_features(self, VFeatureDeg):
        self.degV = 2 * VFeatureDeg
        self.sym_f = self.PolynomialDynamics()
        self.degf = max([Polynomial(i, self.sym_x).TotalDegree() for i in
                         self.sym_f])
        if VFeatureDeg == 1:
            sym_x_expression = np.array([Expression(i) for i in self.sym_x])
            self.sym_phi = np.append(pow(self.sym_x[0], 0), sym_x_expression)
        else:
            self.sym_phi = GetMonomials(self.sym_x, VFeatureDeg)
        self.sym_dphidx = Jacobian(self.sym_phi, self.sym_x)

    def get_features(self, x):
        # x: (num_samples, sys_dim)
        f = self.PolynomialDynamics(sample_states=x)
        n_samples = x.shape[0]
        phi = np.zeros((n_samples, self.sym_phi.shape[0]))
        dphidx = np.zeros((n_samples, self.sym_phi.shape[0], self.num_states))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi[i, :] = [i.Evaluate(env) for i in self.sym_phi]
            dphidx[i, :, :] = [[i.Evaluate(env) for i in j]for j in
                               self.sym_dphidx]
        # np.savez(self.model_file_name + '-' + str(n_samples) +
        #          'samples', phi=phi, dphidx=dphidx, f=f)
        return [phi, dphidx, f]

    def set_levelset_features(self, V, sigma_deg):
        self.sym_V = V
        self.sym_Vdot = self.sym_V.Jacobian(self.sym_x) @ self.sym_f
        self.degVdot = Polynomial(self.sym_Vdot, self.sym_x).TotalDegree()
        deg = int(np.floor((sigma_deg + self.degVdot - self.degV) / 2))
        self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
        self.sym_sigma = GetMonomials(self.sym_x, sigma_deg)
        psi_deg = int(np.floor(max(2 * deg + self.degV, sigma_deg +
                                   self.degVdot) / 2))
        self.sym_psi = GetMonomials(self.sym_x, psi_deg, remove_one=False)

    def get_levelset_features(self, x):
        # x: (num_samples, sys_dim)
        n_samples = x.shape[0]
        V = np.zeros((n_samples, 1))
        Vdot = np.zeros((n_samples, 1))
        xxd = np.zeros((n_samples, 1))
        psi = np.zeros((n_samples, self.sym_psi.shape[0]))
        sigma = np.zeros((n_samples, self.sym_sigma.shape[0]))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            V[i, :] = self.sym_V.Evaluate(env)
            Vdot[i, :] = self.sym_Vdot.Evaluate(env)
            xxd[i, :] = self.sym_xxd.Evaluate(env)
            psi[i, :] = [i.Evaluate(env) for i in self.sym_psi]
            sigma[i, :] = [i.Evaluate(env) for i in self.sym_sigma]
        return [V, Vdot, xxd, psi, sigma]


class VanderPol(ClosedLoopSys):

    def __init__(self):
        self.name = 'VanderPol'
        self.num_states = 2
        self.num_inputs = 0
        self.num_outputs = 0
        self.trueROA = True
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")

    def get_x(self, d=2, num_grid=200):
        x1 = np.linspace(-d, d, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1 = x1[np.nonzero(x1)]
        x2 = x2[np.nonzero(x2)]
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        return np.array([x1, x2])  # (2, num_grid**2)

    def PolynomialDynamics(self, sample_states=None):
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

    def inward_vdp(self, t, y):
        return - np.array([y[1], (1 - y[0]**2) * y[1] - y[0]])

    def outward_vdp(self, t, y):
        return - self.inward_vdp(t, y)

    def step_once(self):
        sol = integrate.RK45(self.inward_vdp, 0, self.states, 1)
        self.states = sol.y

    def SimStableSamples(self, num_samples):
        event = lambda t, x: np.linalg.norm(x) - 15
        event.terminal = True
        x = self.get_x(d=3, num_grid=num_samples)
        stableSamples = np.zeros((2,))
        for i in range(x.shape[-1]):
            sol = integrate.solve_ivp(self.inward_vdp, [0, 15], x[:, i],
                                      events=event)
            # if sol.status == 1:
            # print('event stopping')
            if sol.status == 0 and (np.linalg.norm(sol.y[:, -1])) <= 1e-2:
                stableSamples = np.vstack((stableSamples, x[:, i]))
        return stableSamples.T

    def _phase_portrait(self, ax, ax_max):
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


class TanhPolyCL(ClosedLoopSys):
    """Summary
    # returns the CONTINUOUS TIME closed-loop dynamics of the augmented states,
    # which include the plant state x, the RNN state c, the added two states
    # from the tanh nonlinearity, tau_c, and tau_f
    """

    def __init__(self, CL, model_file_name, taylor_approx=False):
        self.output_kernel = (K.eval(CL.cell.output_kernel))
        self.feedthrough_kernel = (K.eval(CL.cell.feedthrough_kernel))
        self.recurrent_kernel_f = (K.eval(CL.cell.recurrent_kernel_f))
        self.kernel_f = (K.eval(CL.cell.kernel_f))
        self.recurrent_kernel_c = (K.eval(CL.cell.recurrent_kernel_c))
        self.kernel_c = (K.eval(CL.cell.kernel_c))

        self.plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
        self.nx = CL.cell.num_plant_states
        self.units = CL.units
        self.dt = CL.dt

        self.taylor_approx = taylor_approx

        prog = MathematicalProgram()
        self.x = prog.NewIndeterminates(self.nx, "x")
        self.c = prog.NewIndeterminates(self.units, "c")

        if taylor_approx:
            self.sym_x = np.concatenate((self.x, self.c))
            self.num_states = self.nx + self.units
        else:
            self.tau_f = prog.NewIndeterminates(self.units, "tf")
            self.tau_c = prog.NewIndeterminates(self.units, "tc")
            self.sym_x = np.concatenate(
                (self.x, self.c, self.tau_f, self.tau_c))
            self.num_states = self.nx + 3 * self.units
        self.model_file_name = model_file_name

    def InverseRecastMap(self):
        [arg_f, arg_c] = self.ArgsForTanh(self.x, self.c)
        tau_f = [tanh(i) for i in arg_f]
        tau_c = [tanh(i) for i in arg_c]
        env = dict(zip(np.append(self.tau_c, self.tau_f), tau_c + tau_f))
        return env

    def ArgsForTanh(self, x, c):
        arg_f = x@self.kernel_f + c@self.recurrent_kernel_f
        arg_c = x@self.kernel_c + c@self.recurrent_kernel_c
        return [arg_f, arg_c]

    def get_features(self, x):
        # x: (num_samples, sys_dim)
        f = self.NonlinearDynamics(sample_states=x)
        self.verifi_f = self.NonlinearDynamics()
        n_samples = x.shape[0]
        phi = np.zeros((n_samples, self.sym_phi.shape[0]))
        dphidx = np.zeros((n_samples, self.sym_phi.shape[0], self.num_states))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi[i, :] = [i.Evaluate(env) for i in self.sym_phi]
            dphidx[i, :, :] = [[i.Evaluate(env) for i in j]for j in
                               self.sym_dphidx]
        # np.savez(self.model_file_name + '-' + str(n_samples) +
        #          'samples', phi=phi, dphidx=dphidx, f=f)
        return [phi, dphidx, f]

    def NonlinearDynamics(self, sample_states=None):
        if sample_states is None:
            x = self.x
            c = self.c
        else:
            x = sample_states[:, 0:self.nx]
            c = sample_states[:, self.nx:self.nx + self.units]

        shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0
        u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
        xdot = self.plant.xdot(x.T, u.T).T
        ydot = self.plant.ydot(x.T, u.T).T
        # use the argument to approximate the tau:
        [arg_f, arg_c] = self.ArgsForTanh(shift_y_tm1, c)
        if sample_states is None:
            tau_f = np.array([tanh(i) for i in arg_f])
            tau_c = np.array([tanh(i) for i in arg_c])
        else:
            tau_f = np.tanh(arg_f)
            tau_c = np.tanh(arg_c)
        cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
        # TODO: should be y0dot but let's for now assume the two are the same
        # (since currently all y0=zeros)
        f = np.hstack((xdot, cdot))
        return f

    def PolynomialDynamics(self, sample_states=None):
        if self.taylor_approx:
            if sample_states is None:
                x = self.x
                c = self.c
            else:
                x = sample_states[:, 0:self.nx]
                c = sample_states[:, self.nx:self.nx + self.units]

            shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0
            u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
            xdot = self.plant.xdot(x.T, u.T).T
            ydot = self.plant.ydot(x.T, u.T).T
            # use the argument to approximate the tau:
            [tau_f, tau_c] = self.ArgsForTanh(shift_y_tm1, c)
            cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
            # TODO: should be y0dot but let's for now assume the two are the same
            # (since currently all y0=zeros)
            f = np.hstack((xdot, cdot))
            return f
        else:
            if sample_states is None:
                x = self.x
                c = self.c
                tau_f = self.tau_f
                tau_c = self.tau_c
            else:
                x = sample_states[:, 0:self.nx]
                c = sample_states[:, self.nx:self.nx + self.units]
                tau_f = sample_states[:, self.nx +
                                      self.units:self.nx + 2 * self.units]
                tau_c = sample_states[:, self.nx + 2 *
                                      self.units:self.nx + 3 * self.units]

            shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0

            u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
            xdot = self.plant.xdot(x.T, u.T).T
            ydot = self.plant.ydot(x.T, u.T).T
            cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
            # TODO: should be y0dot but let's for now assume the two are the same
            # (since currently all y0=zeros)
            tau_f_dot = (1 - tau_f**2) * self.ArgsForTanh(ydot, cdot)[0]
            tau_c_dot = (1 - tau_c**2) * self.ArgsForTanh(ydot, cdot)[1]
            f = np.hstack((xdot, cdot, tau_f_dot, tau_c_dot))
            return f

    def sampleInitialStatesInclduingTanh(self, num_samples, **kwargs):
        """sample initial states in [x,c,tau_f,tau_c] space. But since really only
        x and c are independent, bake in the relationship that tau_f=tanh(arg_f)
        and tau_c=tanh(arg_c) here. Also return the augmented poly system dyanmcis
        f for the downstream verification analysis.

        Args:
            CL (TYPE): closedcontrolledsystem

        Returns:
            TYPE: Description
        """
        [x, _, _] = self.plant.get_data(
            num_samples, 1, self.units, **kwargs)[0]
        shifted_y = self.plant.get_obs(x) - self.plant.y0
        c = np.random.uniform(-.01, .01, (num_samples, self.units))
        if self.taylor_approx:
            s = np.hstack((x, c))
        else:
            tanh_f = np.tanh(self.ArgsForTanh(shifted_y, c)[0])
            tanh_c = np.tanh(self.ArgsForTanh(shifted_y, c)[1])
            s = np.hstack((x, c, tanh_f, tanh_c))
        return s

    def linearizeTanhPolyCL(self):
        """
        linearize f, which is the augmented (via the change of variable recasting)
        w.r.t. the states x.

        Args:
            x (TYPE): States
            f (TYPE): the augmented dynamics

        Returns:
            TYPE: the linearization, evaluated at zero
        """
        # TODO: for now linearize at zero, need to linearize at plant.x0

        # x = self.sym_x
        # f = self.PolynomialDynamics()
        x = self.x
        c = self.c
        xc = np.concatenate((self.x, self.c))
        f = self.NonlinearDynamics()
        J = Jacobian(f, xc)
        env = dict(zip(xc, np.zeros(xc.shape)))
        A = np.array([[i.Evaluate(env) for i in j]for j in J])
        # print('A  %s' % A)
        print('eig of the linearized A matrix for augmented with tanh poly system %s' % (
            eig(A)[0]))
        S = solve_lyapunov(A.T, -np.eye(xc.shape[0]))
        return A, S
