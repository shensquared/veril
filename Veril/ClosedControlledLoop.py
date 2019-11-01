import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (MathematicalProgram, Polynomial,
                         Expression, SolutionResult,
                         Variables, Solve, Jacobian, Evaluate,
                         RealContinuousLyapunovEquation, Substitute,
                         MosekSolver)

import Plants
from keras import backend as K
import os
from keras.utils import CustomObjectScope
from CustomLayers import JanetController
from keras.models import load_model


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
        'unit' + str(num_units) + 'step' + str(timesteps) + tag + '.h5'

    with CustomObjectScope({'JanetController': JanetController}):
        model = load_model(model_file_name)
    print(model.summary())
    if NNorCL is 'NN':
        return model
    elif NNorCL is 'CL':
        for this_layer in model.layers:
            if hasattr(this_layer, 'cell'):
                return this_layer


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


def batchSim(CL, timesteps, num_samples=10000):
    """return two sets of initial conditions based on the simulated results.
    One set is the stable trajectory and the other set is the unstable one.

    Args:
        CL (TYPE): Description
        timesteps (TYPE): Description
        init (TYPE): Description
        num_samples (TYPE): Description
    """
    init_x_train = np.random.randn(num_samples, CL.cell.num_plant_states)
    init_c = np.zeros((num_samples, CL.cell.units))
    init = [init_x_train, init_c]
    for i in range(timesteps):
        init = call_CLsys(CL, init, num_samples)
    return init


def originalSysInitialV(CL):
    plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    A0 = CL.linearize()
    full_dim = plant.num_states + CL.units
    prog = MathematicalProgram()
    # constant = prog.NewIndeterminates(1, 'constant')
    # basis = [sym.Monomial(constant[0], 0)]
    x = prog.NewIndeterminates(full_dim, "x")

    if not plant.manifold:
        S0 = solve_lyapunov(A0.T, -np.eye(full_dim))
    else:
        # c = prog.NewIndeterminates(CL.units, "c")
        # full_states = np.hstack((x, c))
        # basis = [sym.Monomial(_) for _ in full_states]
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


class augmentedTanhPolySys(object):
    """Summary
    # returns the CONTINUOUS TIME closed-loop dynamics of the augmented states,
    # which include the plant state x, the RNN state c, the added two states
    # from the tanh nonlinearity, tau_c, and tau_f
    Args:
        CL (TYPE): closed-loop system dynamics
        states: default None, meaning everything drake symbolic; otherwise
        expecting np arrays for numerical samples

    Returns:
        all states (plant, controller, plus the recasted ones)
        polynomial dynamics of the all-states
    """

    def __init__(self, CL):
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

    def ArgsForTanh(self, x, c):
        arg_f = x@self.kernel_f + c@self.recurrent_kernel_f
        arg_c = x@self.kernel_c + c@self.recurrent_kernel_c
        return [arg_f, arg_c]

    def symbolicStatesAndDynamics(self):
        prog = MathematicalProgram()
        x = prog.NewIndeterminates(self.nx, "x")
        c = prog.NewIndeterminates(self.units, "c")
        tau_f = prog.NewIndeterminates(self.units, "tf")
        tau_c = prog.NewIndeterminates(self.units, "tc")
        shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0

        u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
        xdot = self.plant.xdot(x, u)
        ydot = self.plant.ydot(x, u)
        cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
        # TODO: should be y0dot but let's for now assume the two are the same
        # (since currently all y0=zeros)
        tau_f_dot = (1 - tau_f**2) * self.ArgsForTanh(ydot, cdot)[0]
        tau_c_dot = (1 - tau_c**2) * self.ArgsForTanh(ydot, cdot)[1]

        augedStates = np.hstack((x, c, tau_f, tau_c))
        augedDynamics = np.hstack((xdot, cdot, tau_f_dot, tau_c_dot))
        return [augedStates, augedDynamics]

    def sampleInitialStatesInclduingTanh(self, num_samples):
        """sample initial states in [x,c,tau_f,tau_c] space. But since really only
        x and c are independent, bake in the relationship that tau_f=tanh(arg_f)
        and tau_c=tanh(arg_c) here. Also return the augmented poly system dyanmcis
        f for the downstream verification analysis.

        Args:
            CL (TYPE): closedcontrolledsystem

        Returns:
            TYPE: Description
        """
        [x, c, _] = self.plant.get_data(num_samples, 1, self.units)[0]
        shifted_y = self.plant.get_obs(x) - self.plant.y0
        tanh_f = np.tanh(self.ArgsForTanh(shifted_y, c)[0])
        tanh_c = np.tanh(self.ArgsForTanh(shifted_y, c)[1])
        return np.hstack((x, c, tanh_f, tanh_c))

    def linearizeAugmentedTanhPolySys(self):
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
        [x, f] = self.symbolicStatesAndDynamics()
        J = Jacobian(f, x)
        env = dict(zip(x, np.zeros(x.shape)))
        A = np.array([[i.Evaluate(env) for i in j]for j in J])
        # print('A  %s' % A)
        print('eig of the linearized A matrix for augmented with tanh poly system %s' % (
            eig(A)[0]))
        S = solve_lyapunov(A.T, -np.eye(x.shape[0]))
        return A, S
