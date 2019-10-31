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


def augmentedTanhPolySys(CL, return_weights=False):
    """Summary
    # returns the CONTINUOUS TIME closed-loop dynamics of the augmented states,
    # which include the plant state x, the RNN state c, the added two states
    # from the tanh nonlinearity, tau_c, and tau_f
    Args:
        CL (TYPE): closed-loop system dynamics

    Returns:
        all states (plant, controller, plus the recasted ones)
        polynomial dynamics of the all-states
        optional return: all the relevant weights

    """

    output_kernel = (K.eval(CL.cell.output_kernel)).T
    feedthrough_kernel = (K.eval(CL.cell.feedthrough_kernel)).T
    recurrent_kernel_f = (K.eval(CL.cell.recurrent_kernel_f)).T
    kernel_f = (K.eval(CL.cell.kernel_f)).T
    recurrent_kernel_c = (K.eval(CL.cell.recurrent_kernel_c)).T
    kernel_c = (K.eval(CL.cell.kernel_c)).T

    plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(plant.num_states, "x")
    c = prog.NewIndeterminates(CL.units, "c")
    tau_f = prog.NewIndeterminates(CL.units, "tf")
    tau_c = prog.NewIndeterminates(CL.units, "tc")

    shift_y_tm1 = plant.get_obs(x) - plant.y0
    # arg_f = kernel_f@shift_y_tm1 + recurrent_kernel_f@c
    # arg_c = kernel_c@shift_y_tm1 + recurrent_kernel_c@c

    u = output_kernel@c + feedthrough_kernel@shift_y_tm1
    xdot = plant.xdot(x, u)
    ydot = plant.ydot(x, u)
    cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / CL.dt
    # TODO: should be y0dot but let's for now assume the two are the same
    # (since currently all y0=zeros)
    tau_f_dot = (1 - tau_f**2) * (kernel_f@ydot + recurrent_kernel_f@cdot)
    tau_c_dot = (1 - tau_c**2) * (kernel_c@ydot + recurrent_kernel_c@cdot)

    augStates = np.hstack((x, c, tau_f, tau_c))
    f = np.hstack((xdot, cdot, tau_f_dot, tau_c_dot))
    if return_weights:
        return [augStates, f, recurrent_kernel_f, kernel_f,
                recurrent_kernel_c, kernel_c]
    else:
        return [augStates, f]


def linearizeAugmentedTanhPolySys(CL):
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
    [x, f] = augmentedTanhPolySys(CL)
    J = Jacobian(f, x)
    env = dict(zip(x, np.zeros(x.shape)))
    A = np.array([[i.Evaluate(env) for i in j]for j in J])
    # print('A  %s' % A)
    print('eig of the linearized A matrix for augmented with tanh poly system %s' % (
        eig(A)[0]))
    S = solve_lyapunov(A.T, -np.eye(x.shape[0]))
    return A, S
