from __future__ import division, print_function
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
me.init_vprinting()
import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
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

kv =1
folder = './'

# kv =.8
# folder = './LB/'

# kv =1.2
# folder = './UB/'

def Dubins_original():
    l=1
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
    # The gains can now be used to define the required input during simulation to stabilize the system. The input $r$ is simply the gain vector multiplied by the error in the state vector from the equilibrium point, $r(t)=K(x_{eq} - x(t))$.
    # ### Feedback the LQR controller for the closed-loop EOM:
    div = - np.array(q)
    # ### Turn the feedback on the small angle $\theta$ to feedback on the $\sin(\theta)$
    div_in_trig = div
    div_in_trig[0] = sin(div[0])
    feedback = np.dot(K, div_in_trig)

    qdot_cl = qdot.subs(dict(zip(f, feedback)))
    # A_cl = sm.matrix2numpy(qdot_cl.jacobian(q).subs(equilibrium_dict),
    #                        dtype=float)

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

    f_cl = sm.expand_trig(qdot_cl).subs(q_to_x_dict)

    return Vr, x, T, f_cl


def get_Vdot(Vr, x, T, f_cl, x_sample_dict, trig_sample_dict):
    phi = Vr.diff((x, 1))
    this_phi = phi.subs(x_sample_dict)
    this_T = T.subs(trig_sample_dict)
    this_f = this_T@f_cl.subs(x_sample_dict)

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

        this_Vdot = get_Vdot(Vr, x, T, f_cl, x_sample_dict, trig_sample_dict)

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
    x, V_num = balancing_V(x, V_num, tol=10)
    np.save(folder + 'samples.npy', x)
    xxd = [(i.T@i) for i in x]
    psi = [get_monomials(i, 2, True) for i in x]
    file = folder + 'Y.npz'
    Y = [np.array(xxd), np.array(psi), V_num]
    np.savez_compressed(file, xxd=Y[0], psi=Y[1], V_num=Y[2])


def load_x_and_y():
    xx = np.load(folder + 'samples.npy')
    Y = np.load(folder + 'Y.npz')
    return [xx, Y['xxd'], Y['psi'], Y['V_num']]
##############################


def right_hand_side(x, t):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.

    """
    arguments = np.hstack((x))     # States, input, and parameters
    dx = np.array(qdot_cl).T[0]
    return dx


def animate_pendulum(t, states, length, filename=None):
    """Animates the n-pendulum and optionally saves it to file.

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take some time.

    Returns
    -------
    fig : matplotlib.Figure
        The figure.
    anim : matplotlib.FuncAnimation
        The animation.

    """
    # the number of pendulum bobs
    numpoints = states.shape[1] // 2

    # first set up the figure, the axis, and the plot elements we want to
    # animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)

    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')

    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
                     cart_width, cart_height, fill=True, color='red',
                     ec='black')
    ax.add_patch(rect)

    # blank line for the pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return time_text, rect, line,

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = np.hstack((states[i, 0], np.zeros((numpoints - 1))))
        y = np.zeros((numpoints))
        for j in np.arange(1, numpoints):
            x[j] = x[j - 1] + length * np.cos(states[i, j])
            y[j] = y[j - 1] + length * np.sin(states[i, j])
        line.set_data(x, y)
        return time_text, rect, line,

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                                   interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    f_name = './link' + str(n) + '/closed-loop.mp4'
    anim.save(f_name, writer='ffmpeg')


def do_anim(x0):
    t = np.linspace(0.0, 7.0, num=500)
    x = odeint(right_hand_side, x0, t)
    anim = animate_pendulum(t, x, 1. / n)


def x_to_originial(x):
    angles = np.arctan2(x[1:2 * n + 1:2], x[2:2 * n + 2:2])
    velocity = x[-n - 1:]
    return np.concatenate([[x[0]], angles, velocity])


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

Vr, x, T, f_cl = Dubins_original()
# sample(300)
get_Y(Vr, x)
Y = load_x_and_y()
[xx, xxd, psi, V] = Y
print(min(V))

# transformed_basis, T = coordinate_ring_transform(psi)
n= 52
print(check_genericity(psi[:n,]))
# Y[2] = transformed_basis
rho, P, Y = solve_SDP_on_samples(Y,n=n, write_to_file=False)

# x0 = x_to_originial(xx[5])
# do_anim(x0)
