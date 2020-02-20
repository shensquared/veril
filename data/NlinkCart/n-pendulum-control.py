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
from veril.sample_variety import coordinate_ring_transform
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import itertools


n = 1


def NlinkedSys(n):
    q = me.dynamicsymbols('q:{}'.format(n + 1))  # Generalized coordinates
    u = me.dynamicsymbols('dq:{}'.format(n + 1))  # Generalized speeds
    f = me.dynamicsymbols('f:{}'.format(n))      # Force applied to the cart
    # m = sm.symbols('m:{}'.format(n + 1))         # Mass of each bob
    # l = sm.symbols('l:{}'.format(n))             # Length of each link
    m = (0.01 / n) * np.ones(n + 1,)
    l = (1. / n) * np.ones(n + 1,)
    g = 9.81
    t = sm.symbols('t')                     # Gravity and time

    I = me.ReferenceFrame('I')  # Inertial reference frame
    O = me.Point('O')           # Origin point
    O.set_vel(I, 0)             # Origin's velocity is zero

    P0 = me.Point('P0')                 # Hinge point of top link
    P0.set_pos(O, q[0] * I.x)           # Set the position of P0
    P0.set_vel(I, u[0] * I.x)           # Set the velocity of P0
    Pa0 = me.Particle('Pa0', P0, m[0])  # Define a particle at P0

    frames = [I]                              # List to hold the n + 1 frames
    points = [P0]                             # List to hold the n + 1 points
    # List to hold the n + 1 particles
    particles = [Pa0]
    # List to hold the n + 1 applied forces, including the input force, f
    forces = [(P0, f[0] * I.x - m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's

    for i in range(n):
        # Create a new frame
        Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])
        # Set angular velocity
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        # Add it to the frames list
        frames.append(Bi)

        # Create a new point
        Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)
        # Set the velocity
        Pi.v2pt_theory(points[-1], I, Bi)
        # Add it to the points list
        points.append(Pi)

        # Create a new particle
        Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])
        # Add it to the particles list
        particles.append(Pai)

        # Set the force applied at the point
        forces.append((Pi, -m[i + 1] * g * I.y))
        # Define the kinematic ODE:  dq_i / dt - u_i = 0
        kindiffs.append(q[i + 1].diff(t) - u[i + 1])
    for i in range(n - 1):
        forces.append((frames[i + 1], f[i + 1] * frames[i].z))

    # Initialize the object
    kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    # Generate EoM's fr + frstar = 0
    fr, frstar = kane.kanes_equations(forces, particles)

    equilibrium_point = [sm.S(0)] + [sm.pi / 2] * \
        (len(q) - 1) + [sm.S(0)] * len(u)
    equilibrium_dict = dict(zip(q + u, equilibrium_point))
    print('orig coordiante fixed pt %s' % equilibrium_dict)

    M, F_A, F_B, r = kane.linearize(new_method=True, op_point=equilibrium_dict)

    M_num = sm.matrix2numpy(M, dtype=float)
    F_A_num = sm.matrix2numpy(F_A, dtype=float)
    F_B_num = sm.matrix2numpy(F_B, dtype=float)
    A = np.linalg.solve(M_num, F_A_num)
    B = np.linalg.solve(M_num, F_B_num)

    # Also convert `equilibrium_point` to a numeric array:
    equilibrium_point = np.asarray([x.evalf()
                                    for x in equilibrium_point], dtype=float)
    # Now that we have a linear system, the SciPy package can be used to design an optimal controller for the system.
    # So now we can compute the optimal gains with a linear quadratic
    # regulator. I chose identity matrices for the weightings for simplicity.
    from scipy.linalg import solve_continuous_are
    Q = np.eye(A.shape[0])
    R = np.eye(B.shape[1])
    S = solve_continuous_are(A, B, Q, R)
    K = np.dot(np.dot(np.linalg.inv(R), B.T),  S)
    # The gains can now be used to define the required input during simulation to stabilize the system. The input $r$ is simply the gain vector multiplied by the error in the state vector from the equilibrium point, $r(t)=K(x_{eq} - x(t))$.
    # ### Feedback the LQR controller for the closed-loop EOM:
    states = np.array(q + u)
    div = equilibrium_point - states
    # ### Turn the feedback on the small angle $\theta$ to feedback on the $\sin(\theta)$

    from sympy import sin, cos
    div_in_trig = div
    for i in range(n):
        div_in_trig[i + 1] = sin(div[i + 1])
    feedback = np.dot(K, div_in_trig)

    # List to hold the n + 1 applied forces, including the input force, f
    forces_fb = [(P0, feedback[0] * I.x - m[0] * g * I.y)]

    for i in range(n):
        # Set the force applied at the point
        forces_fb.append((points[i + 1], -m[i + 1] * g * I.y))
    for i in range(n - 1):
        forces_fb.append((frames[i + 1], feedback[i + 1] * frames[i].z))

    kane_cl = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    fr_cl, frstar_cl = kane_cl.kanes_equations(forces_fb, particles)

    full_mass = (kane_cl.mass_matrix_full)
    full_force = (kane_cl.forcing_full)

    # ### Double check the EOM is in terms of the trigs and velocities only
    # full_mass_num = full_mass.subs(equilibrium_dict)
    # full_force_num = full_force.subs(equilibrium_dict)
    # ### Double check the CL dynamics in terms of $q$ $\dot q$ are stable
    M_cl, F_A_cl, F_B_cl, _ = kane_cl.linearize(new_method=True,
                                                op_point=equilibrium_dict)
    M_num_cl = sm.matrix2numpy(M_cl, dtype=float)
    F_A_num_cl = sm.matrix2numpy(F_A_cl, dtype=float)
    A_cl = np.linalg.solve(M_num_cl, F_A_num_cl)

    print('closed-loop A eigs %s' % np.linalg.eig(A_cl)[0])
    from scipy.linalg import solve_lyapunov
    P = solve_lyapunov(A_cl.T, -np.eye(A_cl.shape[0]))
    V = sm.expand_trig((div_in_trig).T@P@(div_in_trig))
    sm.N(V.subs(equilibrium_dict))

# def recat_state(n):
    sins = me.dynamicsymbols('s:{}'.format(n + 1))[1:]
    coss = me.dynamicsymbols('c:{}'.format(n + 1))[1:]
    x = [q[0]]
    for i in range(n):
        x = x + [sins[i]] + [coss[i]]
    x += u

    x0 = [sm.S(0)] + [sm.S(1), sm.S(0)] * (n) + [sm.S(0)] * (n + 1)
    x0_dict = dict(zip(x, x0))
    # return x, x0_dict


# def recast_transform(n):
    T = sm.Matrix(np.zeros((3 * n + 2, 2 * n + 2)))
    T[0, 0] = sm.S(1)
    for i in range(n):
        T[2 * i + 1, i + 1] = coss[i]
        T[2 * i + 2, i + 1] = - sins[i]
    for i in range(n + 1):
        T[-i - 1, -i - 1] = sm.S(1)
    # return T


# def recast_sym_subs_dict(q):
    sin_q = [sin(i) for i in q[1:]]
    cos_q = [cos(i) for i in q[1:]]
    key = sin_q + cos_q
    values = sins + coss
    q_to_x_dict = dict(zip(key, values))
    # return q_to_x_dict

    Vr = V.subs(q_to_x_dict)
    print('Vr at the fixed pt %s' % Vr.subs(x0_dict))


# recast the mass and force matrices
    M = full_mass.subs(q_to_x_dict)
    F = sm.expand_trig(full_force).subs(q_to_x_dict)

# M_num = M.subs(x0_dict)
# F_num = F.subs(x0_dict)


# debug: should match the original form
# assert(full_mass_num == M_num)
# assert(full_force_num == F_num)
# Create a callable function to evaluate the mass matrix
    M_func = sm.lambdify(states, kane_cl.mass_matrix_full)
    f_func = sm.lambdify(states, kane_cl.forcing_full)

    return M, F, T, Vr, x, M_func, f_func


def get_Vdot(M, F, T, Vr, x, x_sample_dict, trig_sample_dict):
    phi = Vr.diff((x, 1))
    this_phi = phi.subs(x_sample_dict)

    this_mass = sm.matrix2numpy(M.subs(trig_sample_dict), dtype=float)
    this_mass_inv = np.linalg.inv(this_mass)

    this_force = F.subs(x_sample_dict)
    this_T = T.subs(trig_sample_dict)
    this_f = this_T@this_mass_inv@this_force

    this_Vdot = this_phi@sm.matrix2numpy(this_f)
    return this_Vdot


def sample(system_params, n_samples):
    M, F, T, Vr, x, M_func, f_func = system_params
    x_samples = []
    while len(x_samples) < n_samples:

        # alpha = np.random.randn(3 * n + 2)
        # beta = np.random.randn(3 * n + 2)
        alpha = np.random.uniform(-1, 1, 3 * n + 2)
        beta = np.random.uniform(-1 / 2, 1 / 2, 3 * n + 2)
        t = sm.Symbol('t', real=True)
        x_sample_dict = dict(zip(x, alpha * t + beta))

        angle = np.pi / 2 + np.random.uniform(-np.pi / 2, np.pi / 2, n)
        trig_sample_dict = dict(zip(x[1:2 * n + 1:2], np.sin(angle)))
        trig_sample_dict.update(dict(zip(x[2:2 * n + 2:2], np.cos(angle))))

        x_sample_dict.update(trig_sample_dict)

        this_Vdot = get_Vdot(M, F, T, Vr, x, x_sample_dict, trig_sample_dict)

        t_num = sm.solve(this_Vdot, t)

        if len(t_num) > 0:
            for i in t_num:
                sol_dict = dict(zip(x, alpha * i + beta))
                sol_dict.update(trig_sample_dict)
                xx = np.array(list(sol_dict.values()), dtype='float')
                print(xx)
                x_samples.append(xx)
                file = './link' + str(n) + '/x_samples'
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
    folder = './link' + str(n)
    x = np.load(folder + '/samples.npy')
    xxd = [(i.T@i) for i in x]
    psi = [get_monomials(i, 2, True) for i in x]
    V_num = [V.subs(dict(zip(states, i))) for i in x]
    file = './link' + str(n) + '/Y.npz'
    Y = [np.array(xxd), np.array(psi), np.array(V_num, dtype=float)]
    np.savez_compressed(file, xxd=Y[0], psi=Y[1], V_num=Y[2])


def load_x_and_y():
    folder = './link' + str(n)
    xx = np.load(folder + '/samples.npy')
    Y = np.load('./link' + str(n) + '/Y.npz')
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
    dx = np.array(np.linalg.solve(M_func(*arguments), f_func(*arguments))).T[0]
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
    x =np.array([orig[0]])
    for i in range(n):
        tt = orig[i+1]
        x = np.concatenate([x, [np.sin(tt)]])
        x = np.concatenate([x, [np.cos(tt)]])
    velocity = orig[-n - 1:]
    return np.concatenate([x, velocity])


def solve_SDP_on_samples(Vr, Y, write_to_file=False):
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

    for i in range(psi.shape[0]):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        prog.AddConstraint(residual == 0)

    prog.AddCost(-rho)
    solver = MosekSolver()
    log_file = "sampling_variety_SDP.text" if write_to_file else ""
    solver.set_stream_logging(False, log_file)
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    P = result.GetSolution(P)
    rho = result.GetSolution(rho)
    # V = system.sym_V / rho
    # TODO: double check if scaling at this stage affects the downstream
    # oversampling
    print(rho)
    return Vr, rho, P, Y

system_params = NlinkedSys(n)
M, F, T, Vr, x, M_func, f_func = system_params

# sample(system_params, 100)
# get_Y(Vr, x)
# x0 = np.hstack((
#     -2,
#     (np.pi / 2) * np.ones(n) - np.random.uniform(-1, 1, n),
#     1 * np.ones(n + 1)))
Y = load_x_and_y()
[xx, xxd, psi, V] = Y

# transformed_basis, T = coordinate_ring_transform(psi)
# solve_SDP_on_samples(Vr, Y, write_to_file=False)s


x0 = x_to_originial(xx[np.argmin(V)])
x = original_to_x(x0)

do_anim(x0)


# random_states = [.6, -1.7, -.1, .1]
# ss = recast_num_from_x(random_states)
# print(Vr.subs(dict(zip(x, ss))))
# x = sim_once(random_states)
