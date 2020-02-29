from __future__ import division, print_function
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
me.init_vprinting()
# import sys
# sys.path.append(
#     "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from pydrake.all import (Polynomial, Variable, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)
from veril.sample_variety import coordinate_ring_transform, check_genericity, balancing_V
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import itertools
import os
from sympy import sin, cos

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import matplotlib.lines as mlines


def Dubins_original():
    l = 1
    q = me.dynamicsymbols('q:{}'.format(3))  # Generalized coordinates
    f = me.dynamicsymbols('f:{}'.format(2))      # Force applied to the cart
    t = sm.symbols('t')                     # Gravity and time

    equilibrium_point = np.zeros(5,)
    equilibrium_dict = dict(zip(q + f, equilibrium_point))
    print('orig coordiante fixed pt %s' % equilibrium_dict)

    u = [f[0] + kv * l, f[1] + l]

    qdot = sm.Matrix([u[0] - kv * l,
                      u[0] * q[-1] + u[1] - l * sm.cos(q[0]),
                      -u[0] * q[1] + l * sm.sin(q[0])])

    # A = sm.matrix2numpy(qdot.jacobian(q).subs(equilibrium_dict), dtype=float)
    # B = sm.matrix2numpy(qdot.jacobian(f).subs(equilibrium_dict), dtype=float)

    # M_num = sm.matrix2numpy(M, dtype=float)
    # F_A_num = sm.matrix2numpy(F_A, dtype=float)
    # F_B_num = sm.matrix2numpy(F_B, dtype=float)

    # A = np.linalg.solve(M_num, F_A_num)
    # B = np.linalg.solve(M_num, F_B_num)

    # Now that we have a linear system, the SciPy package can be used to design an optimal controller for the system.
    # So now we can compute the optimal gains with a linear quadratic
    # regulator. I chose identity matrices for the weightings for simplicity.
    # from scipy.linalg import solve_continuous_are
    # Q = np.eye(A.shape[0])
    # R = np.eye(B.shape[1])
    # S = solve_continuous_are(A, B, Q, R)
    # K = np.dot(np.dot(np.linalg.inv(R), B.T),  S)
    # print('saving gain')
    K = np.load('gain.npy')
    # K = np.ones((2, 3))
    # The gains can now be used to define the required input during simulation to stabilize the system. The input $r$ is simply the gain vector multiplied by the error in the state vector from the equilibrium point, $r(t)=K(x_{eq} - x(t))$.
    # ### Feedback the LQR controller for the closed-loop EOM:
    div = - np.array(q)
    # ### Turn the feedback on the small angle $\theta$ to feedback on the $\sin(\theta)$
    div_in_trig = div
    div_in_trig[0] = sin(div[0])
    feedback = np.dot(K, div_in_trig)

    qdot_cl = sm.expand_trig(sm.simplify(qdot.subs(dict(zip(f, feedback)))))
    # A_cl = sm.matrix2numpy(qdot_cl.jacobian(q).subs(equilibrium_dict),
    # dtype=float)

    # print('closed-loop A eigs %s' % np.linalg.eig(A_cl)[0])
    # from scipy.linalg import solve_lyapunov
    # P = solve_lyapunov(A_cl.T, -np.eye(A_cl.shape[0]))
    # np.save('P.npy', P)
    P = np.load('P.npy')
    V = sm.expand_trig((div_in_trig).T@P@(div_in_trig))
    sm.N(V.subs(equilibrium_dict))

# def recat_state(n):
    sins = me.dynamicsymbols('s:{}'.format(1))
    coss = me.dynamicsymbols('c:{}'.format(1))
    x = sins + coss + q[1:]

    x0 = [0, 1, 0, 0]
    x0_dict = dict(zip(x, x0))
    # return x, x0_dict


# def recast_transform(n):
    T = sm.Matrix(np.zeros((4, 3)))
    for i in range(2):
        T[-i - 1, -i - 1] = sm.S(1)
    T[0, 0] = coss[0]
    T[1, 0] = - sins[0]
    # return T


# def recast_sym_subs_dict(q):
    sin_q = [sin(q[0])]
    cos_q = [cos(q[0])]
    key = sin_q + cos_q
    values = sins + coss
    q_to_x_dict = dict(zip(key, values))
    # return q_to_x_dict

    Vr = V.subs(q_to_x_dict)
    print('Vr at the fixed pt %s' % Vr.subs(x0_dict))

    f_cl = sm.simplify(T@sm.expand_trig(qdot_cl).subs(q_to_x_dict))

    return Vr, x, f_cl, qdot_cl, q


def get_Vdot(Vr, x, f_cl, x_sample_dict, trig_sample_dict):
    phi = Vr.diff((x, 1))
    this_phi = phi.subs(x_sample_dict)
    # this_T = T.subs(trig_sample_dict)
    # this_f = this_T@f_cl.subs(x_sample_dict)
    this_f = f_cl.subs(x_sample_dict)

    this_Vdot = this_phi@sm.matrix2numpy(this_f)
    return this_Vdot


def sample(n_samples):
    x_samples = []
    while len(x_samples) < n_samples:

        # alpha = np.random.randn(3 * n + 2)*10
        # beta = np.random.randn(3 * n + 2)
        alpha = np.random.uniform(-5, 5, 4)
        beta = np.random.uniform(-1, 1, 4)
        # beta = np.zeros(3 * n + 2)
        t = sm.Symbol('t', real=True)
        x_sample_dict = dict(zip(x, alpha * t + beta))

        angle = np.random.uniform(-np.pi, np.pi, 1)
        trig_sample_dict = dict(zip([x[0]], [np.sin(angle)]))
        trig_sample_dict.update(dict(zip([x[1]], [np.cos(angle)])))

        x_sample_dict.update(trig_sample_dict)

        this_Vdot = get_Vdot(Vr, x, f_cl, x_sample_dict, trig_sample_dict)

        t_num = sm.solve(this_Vdot, t)

        if len(t_num) > 0:
            for i in t_num:
                sol_dict = dict(zip(x, alpha * i + beta))
                sol_dict.update(trig_sample_dict)
                xx = np.array(list(sol_dict.values()), dtype='float')
                print(xx)
                file = folder + 'x_samples.npy'
                if os.path.exists(file):
                    x_samples = np.load(file)
                    x_samples = np.vstack((x_samples, xx))
                else:
                    x_samples.append(xx)
                np.save(file, np.array(x_samples))
        else:
            print('no root')
    return x_samples


def get_monomials(x, deg, rm_one):
    c = 1
    _ = itertools.combinations_with_replacement(np.append(c, x), deg)
    basis = [np.prod(j) for j in _]
    if rm_one:
        basis = basis[1:]
    return (basis[:: -1])


def get_Y(V, states):
    x = np.load(folder + 'x_samples.npy')
    x = x[~np.any(np.abs(x) > 20, axis=1)]
    V_num = np.array([V.subs(dict(zip(states, i))) for i in x], dtype=float)
    x, V_num = balancing_V(x, V_num, tol=25)
    np.save(folder + 'samples.npy', x)
    xxd = [(i.T@i) for i in x]
    psi = [get_monomials(i, 3, False) for i in x]
    file = folder + 'Y.npz'
    Y = [np.array(xxd), np.array(psi), V_num]
    np.savez_compressed(file, xxd=Y[0], psi=Y[1], V_num=Y[2])


def load_x_and_y():
    xx = np.load(folder + 'samples.npy')
    Y = np.load(folder + 'Y.npz')
    return [xx, Y['xxd'], Y['psi'], Y['V_num']]
##############################


def right_hand_side(x, t, args):
    dx = sm.matrix2numpy(qdot_cl.subs(dict(zip(q, x))), dtype=float).T[0]
    if args: dx = -dx
    return dx


def x_to_originial(x):
    x = x.reshape((-1, 4))
    angles = np.arctan2(x[:, 0], x[:, 1]).reshape((-1, 1))
    velocity = x[:, -2:]
    return np.hstack([angles, velocity])


def original_to_x(orig):
    x = [orig[0]]
    for i in range(n):
        x.append(np.sin(orig[i + 1]))
        x.append(np.cos(orig[i + 1]))
    velocity = orig[-n - 1:]
    return np.concatenate([x, velocity])


def solve_SDP_on_samples(Y, n=None, write_to_file=False):
    prog = MathematicalProgram()
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    [xx, xxd, psi, V] = Y
    # print('SDP V %s' % V)
    num_samples, dim_psi = psi.shape
    # print('num_samples is %s' % num_samples)
    print('dim_psi is %s' % dim_psi)
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)
    if n is not None:
        num_samples = n
    for i in range(num_samples):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        prog.AddConstraint(residual == 0)

    prog.AddCost(-rho)
    solver = MosekSolver()
    log_file = "sampling_variety_SDP.text" if write_to_file else ""
    solver.set_stream_logging(True, log_file)
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    P = result.GetSolution(P)
    rho = result.GetSolution(rho)
    print(rho)
    return rho, P, Y


def scatter_3d(Vr, rhos, x, r_max=[1.2, 1.2], res=50):
    [rho1, rho2, rho3] = rhos
    theta = np.linspace(-np.pi, np.pi, res)
    X = np.linspace(-r_max[0], r_max[0], res)
    Y = np.linspace(-r_max[1], r_max[1], res)
    X, Y, Z = np.meshgrid(theta, X, Y)
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    sins = np.sin(X)
    coss = np.cos(X)
    samples = np.array([sins, coss, Y, Z]).T
    np.save('Densedensex.npy', samples)
    vdots = [get_Vdot(Vr, x, f_cl, dict(zip(x, i)), dict(zip(x, i))) for i in
             samples]
    np.save('Vdot_values', vdots)
    # values = np.array([Vr.subs(dict(zip(x, i))) for i in samples])
    # np.save('V_values', values)
    # values = np.load('V_values.npy', allow_pickle=True)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('angle')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    [X1, Y1, Z1] = X[values <= rho1], Y[values <= rho1], Z[values <= rho1]
    ax.scatter(X1, Y1, Z1, c='red')

    idx2 = (values >= rho1) & (values <= rho2)
    ax.scatter(X[idx2], Y[idx2], Z[idx2], c='blue', alpha=.2)

    idx3 = (values >= rho2) & (values <= rho3)
    ax.scatter(X[idx3], Y[idx3], Z[idx3], c='yellow', alpha=.1)

    plt.show()


import plotly.graph_objects as go


def scatter_volume(rhos, r_max=[1.2, 1.2], res=50):
    [rho1, rho2, rho3] = rhos
    theta = np.linspace(-np.pi, np.pi, res)
    X = np.linspace(-r_max[0], r_max[0], res)
    Y = np.linspace(-r_max[1], r_max[1], res)
    X, Y, Z = np.meshgrid(theta, X, Y)
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
#     sins = np.sin(X)
#     coss = np.cos(X)
    # values = np.array([Vr.subs(dict(zip(x, i))) for i in samples])
    # np.save('V_values', values)
    values = np.load('V_values.npy')
    # values = np.load('Vdot_values.npy')
    values[values <= rho1] = rho1
    allbad = np.load('all_bad.npy')
    bad_samples = go.Scatter3d(
        x=allbad[:, 0], y=allbad[:, 1], z=allbad[:, 2], mode='markers')
    roa = go.Volume(
        x=X, y=Y, z=Z,
        value=values.flatten(),
        isomin=rho1,
        isomax=rho3,
        opacity=0.2,
        surface_count=3,
        opacityscale=[[rho1, 1], [rho2, 0.3], [rho3, .1]],
    )
    folder = ['./', './LB/', './UB/']
    # bad_traj = np.load(folder + 'bad_traj.npy')
    traj = [go.Scatter3d(
        x=np.load(i + 'bad_traj.npy')[:, 0],
        y=np.load(i + 'bad_traj.npy')[:, 1],
        z=np.load(i + 'bad_traj.npy')[:, 2],
        marker=dict(
            size=1,
            # color=z,
            # colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=1
        ),
    ) for i in folder ]

    # scene_xaxis_showticklabels=True,
    # scene_yaxis_showticklabels=Fals,
#                   scene_zaxis_showticklabels=False)
    # theta = allbad[:,0] + np.linspace(-.05, .05, 15)
    # X = allbad[:,1] + np.linspace(-.05, .05, 15)
    # Y = allbad[:,2] + np.linspace(-.05, .05, 15)
    # X, Y, Z = np.meshgrid(theta, X, Y)
    # X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

    # samples = np.array([X , Y, Z]).T
    # field = [qdot_cl.subs(dict(zip(q,i))) for i in samples]
    # [U,V,W] = np.array(field,dtype=float).T
    # cones = dict(type='cone',x=X, y=Y, z=Z, u=U, v=V, w=W,
    #               sizemode='scaled',
    #               sizeref=3,
    #               showscale=True,
    #               # colorscale=pl_curl,
    #               # colorbar=dict(thickness=20, ticklen=4, len=0.75),
    #               anchor='tail'
    #           )
    # fig = go.Figure(data=[roa, bad_samples, cones])
    fig = go.Figure(data=[roa, bad_samples] + traj)
    # fig.update_layout(layout_showscale=False)
    fig.show()


def sim_once(x0=None, backwards=False, t_max=10):
    if x0 is None:
        x0 = x_to_originial(np.load(folder + 'bad_sample.npy'))
    x0 = x0.ravel()
    t = np.linspace(0.0, t_max, num=500)
    x = odeint(right_hand_side, x0, t, args=(backwards,))

    lines = plt.plot(t, x)
    lab = plt.xlabel('Time [sec]')
    plt.show()
    return x


# def sim_samples(r_max=[1.2, 1.2], res=50):
#     # theta = np.linspace(-np.pi, np.pi, res)
#     # X = np.linspace(-r_max[0], r_max[0], res)
#     # Y = np.linspace(-r_max[1], r_max[1], res)
#     # X, Y, Z = np.meshgrid(theta, X, Y)
#     # X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
#     t = np.linspace(0.0, 10.0, num=500)
#     # samples = np.array([X,Y,Z]).T
#     # np.save('dense_samples', samples)
#     samples = np.load('dense_samples.npy')
#     x = [odeint(right_hand_side, i, t)[-1] for i in samples]
#     np.save('final', x)


def right_hand_side_4d(x0, t):
    return sm.matrix2numpy(f_cl.subs(dict(zip(x, x0))), dtype=float).T[0]


def sim_4d_once():
    # x0 = [np.sin(-np.pi / 3), np.cos(-np.pi / 3), 0, 0]
    x0 = np.load('bad_sample.npy')
    t = np.linspace(0.0, 10.0, num=500)
    x = odeint(right_hand_side_4d, x0, t)
    lines = plt.plot(t, x)
    lab = plt.xlabel('Time [sec]')
    plt.legend(str(kv))
    plt.show()
    return x


def solve_fixed_pt(f_cl, x):
    first = sm.Matrix(sm.solve(f_cl, x)[-1])
    unit = first[0]**2 + first[1]**2 - 1
    aa = sm.solve(unit, x)
    bb = [first.subs(dict(zip(x, i))) for i in aa]
    return aa


def traj_to_unstable():
    x0 = x_to_originial(np.load(folder + 'bad_sample.npy'))
    x0 = x0.ravel()
    A = qdot_cl.jacobian(q).subs(dict(zip(q, x0)))
    A = sm.matrix2numpy(A, dtype=float)
    # eigs = np.linalg.eig(A)
    x1 = x0 - A@x0 * 1e-4
    # starting from x1 (close to the unstable pt), roll-out backwards
    unstable_out = sim_once(x0=x1, backwards=True, t_max=12)
    # double check that starting from the backward roll-out traj end pt, can
    # get to the unstable fixed pt
    to_unstable = sim_once(x0=unstable_out[-1])
    # save the to_unstable traj
    np.save(folder + 'bad_traj.npy', to_unstable)

# kv = 1
# folder = './'
# n = 78

kv = .8
folder = './LB/'
n = 66

# kv =1.2
# folder = './UB/'
# n=76


Vr, x, f_cl, qdot_cl, q = Dubins_original()

# solve_fixed_pt(f_cl,x)
# sample(300)
# get_Y(Vr, x)


# Y = load_x_and_y()
# [xx, xxd, psi, V] = Y
# transformed_basis, T = coordinate_ring_transform(psi, True, False)
# print(check_genericity(psi[:n, ], False))
# Y[2] = transformed_basis
# rho, P, Y = solve_SDP_on_samples(Y, n=n, write_to_file=False)


# rhos = [.367, 0.57, 0.633]
# scatter_3d(Vr, rhos, x)
# scatter_volume(rhos)
# vector_filed_plot(r_max=[1.2, 1.2], res=10)
# x3d = sim_once()
# x4d = sim_4d_once()
# sim_samples(r_max=[1.2, 1.2], res=10)

# traj_to_unstable()
rhos = [.367, 0.57, 0.633]
scatter_volume(rhos)
