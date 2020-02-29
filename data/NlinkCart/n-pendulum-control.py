from __future__ import division, print_function
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
me.init_vprinting()
from pydrake.all import MosekSolver, MathematicalProgram
from veril.sample_variety import coordinate_ring_transform, check_genericity
from scipy.integrate import odeint
import itertools
import os
from scipy import integrate
import cvxpy as cp


def NlinkedSys(n):
    q = me.dynamicsymbols('q:{}'.format(n + 1))  # Generalized coordinates
    u = me.dynamicsymbols('dq:{}'.format(n + 1))  # Generalized speeds
    f = me.dynamicsymbols('f:{}'.format(n))      # Force applied to the cart

    # q = sm.symbols('q:{}'.format(n + 1))  # Generalized coordinates
    # u = sm.symbols('dq:{}'.format(n + 1))  # Generalized speeds
    # f = sm.symbols('f:{}'.format(n))

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
    # print('orig coordiante fixed pt %s' % equilibrium_dict)

 ##########################################################################
    try:
        K = np.load(folder + '/gain.npy')
        print('loading gain')
    except:
        M, F_A, F_B, r = kane.linearize(
            new_method=True, op_point=equilibrium_dict)

        M_num = sm.matrix2numpy(M, dtype=float)
        F_A_num = sm.matrix2numpy(F_A, dtype=float)
        F_B_num = sm.matrix2numpy(F_B, dtype=float)
        A = np.linalg.solve(M_num, F_A_num)
        B = np.linalg.solve(M_num, F_B_num)

        # Now that we have a linear system, the SciPy package can be used to design an optimal controller for the system.
        # So now we can compute the optimal gains with a linear quadratic
        # regulator. I chose identity matrices for the weightings for
        # simplicity.
        from scipy.linalg import solve_continuous_are
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1])
        S = solve_continuous_are(A, B, Q, R)
        K = np.dot(np.dot(np.linalg.inv(R), B.T),  S)
        print('saving gain')
        np.save(folder + '/gain.npy', K)
     ##########################################################################
    # The gains can now be used to define the required input during simulation to stabilize the system. The input $r$ is simply the gain vector multiplied by the error in the state vector from the equilibrium point, $r(t)=K(x_{eq} - x(t))$.
    # ### Feedback the LQR controller for the closed-loop EOM:
    # # Also convert `equilibrium_point` to a numeric array:
    x0 = np.asarray([x.evalf() for x in equilibrium_point], dtype=float)
    states = np.array(q + u)
    div = x0 - states
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
    try:
        P = np.load(folder + '/P.npy')
        print('loading P')
    except:
        M_cl, F_A_cl, F_B_cl, _ = kane_cl.linearize(new_method=True,
                                                    op_point=equilibrium_dict)
        M_num_cl = sm.matrix2numpy(M_cl, dtype=float)
        F_A_num_cl = sm.matrix2numpy(F_A_cl, dtype=float)
        A_cl = np.linalg.solve(M_num_cl, F_A_num_cl)

        # print('closed-loop A eigs %s' % np.linalg.eig(A_cl)[0])
        from scipy.linalg import solve_lyapunov
        P = solve_lyapunov(A_cl.T, -np.eye(A_cl.shape[0]))
        print('saving P')
        np.save(folder + '/P.npy', P)

    V = sm.expand_trig((div_in_trig).T@P@(div_in_trig))
    sm.N(V.subs(equilibrium_dict))

# def recat_state(n):
    # sins = me.dynamicsymbols('s:{}'.format(n + 1))[1:]
    # coss = me.dynamicsymbols('c:{}'.format(n + 1))[1:]

    sins = sm.symbols('s:{}'.format(n + 1))[1:]
    coss = sm.symbols('c:{}'.format(n + 1))[1:]

    x = [q[0]]
    for i in range(n):
        x = x + [sins[i]] + [coss[i]]
    x += u

    x0_dict = dict(zip(x, [0] + [1, 0] * (n) + [0] * (n + 1)))

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

    return [M, F, T, Vr, x, M_func, f_func, q, u, x0]


def get_Vdot(M, F, T, Vr, x, x_sample_dict, trig_sample_dict):
    phi = sm.Matrix(Vr.diff((x, 1)))
    phi_T = phi.T@T
    this_phi_T = phi_T.subs(x_sample_dict)

    this_mass = sm.matrix2numpy(M.subs(trig_sample_dict), dtype=float)
    this_mass_inv = np.linalg.inv(this_mass)

    this_force = F.subs(x_sample_dict)
    this_f = this_mass_inv@this_force

    this_Vdot = this_phi_T@sm.matrix2numpy(this_f)
    return this_Vdot


def sample(system_params, n_samples):
    M, F, T, Vr, x, M_func, f_func, q, u, x0 = system_params
    x_samples = []
    t = sm.Symbol('t', real=True)
    while len(x_samples) < n_samples:
        alpha = np.random.randn(n + 2)
        beta = np.random.randn(n + 2)
        # alpha = np.random.uniform(-5, 5, 3 * n + 2)
        # beta = np.random.uniform(-1, 1, 3 * n + 2)
        excluded_angles = [x[0], *x[-n - 1:]]
        x_sample_dict = dict(zip(excluded_angles, alpha * t + beta))

        ang = np.random.uniform(-np.pi, np.pi, n)
        trigs_angle = np.array([[i, j] for (i, j) in zip(np.sin(ang), np.cos(ang))]).ravel()
        trig_sample_dict = dict(zip(x[1:2 * n + 1], trigs_angle))
        # trig_sample_dict.update(dict(zip(x[2:2 * n + 2:2], np.cos(ang))))

        x_sample_dict.update(trig_sample_dict)

        this_Vdot = get_Vdot(M, F, T, Vr, x, x_sample_dict, trig_sample_dict)

        t_num = sm.solve(this_Vdot, t)

        if len(t_num) > 0:
            for i in t_num:
                xx = alpha * i + beta
                xx = np.insert(xx, 1, trigs_angle)
                print(xx)
                # double checking the root is accurate
                # sol_dict = dict(zip(x, xx))
                # print(get_Vdot(M, F, T, Vr, x, sol_dict, sol_dict))
                file = './link' + str(n) + '/x_samples.npy'
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
    _ = itertools.combinations_with_replacement(np.append(1, x), deg)
    basis = [np.prod(j) for j in _]
    if rm_one:
        basis = basis[1:]
    return (basis[:: -1])


def get_Y(V, states):
    folder = './link' + str(n)
    x = np.load(folder + '/x_samples.npy')
    print('raw roots number' + str(x.shape[0]))
    x = x[~np.any(np.abs(x) > 30, axis=1)]
    np.save(folder + '/samples.npy', x)
    xxd = [(i.T@i) for i in x]
    psi = [get_monomials(i, 2, True) for i in x]
    V_num = [V.subs(dict(zip(states, i))) for i in x]
    file = folder + '/Y.npz'
    Y = [np.array(xxd), np.array(psi), np.array(V_num, dtype=float)]
    np.savez_compressed(file, xxd=Y[0], psi=Y[1], V_num=Y[2])


def load_x_and_y():
    xx = np.load(folder + '/samples.npy')
    Y = np.load(folder + '/Y.npz')
    return [xx, Y['xxd'], Y['psi'], Y['V_num']]
##############################


def right_hand_side(t, x):
    """Returns the derivatives of the states.
    """
    arguments = np.hstack((x))     # States, input, and parameters
    dx = np.array(np.linalg.solve(M_func(*arguments), f_func(*arguments))).T[0]
    return dx


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


def solve_SDP_on_samples(Y, write_to_file=False):
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


def solve_SDP_on_samples_CVX(Y, write_to_file=False):
    [xx, xxd, psi, V] = Y
    num_samples, monomial_dim = psi.shape
    constraints = []
    P = cp.Variable((monomial_dim, monomial_dim), symmetric=True)
    rho = cp.Variable(1)
    constraints += [P >> 0]
    constraints += [rho >= 0]
    for i in range(num_samples):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        constraints += [residual == 0]

    prob = cp.Problem(cp.Minimize(-rho), constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    P = P.value
    rho = rho.value
    print('rho value %s' % rho)
    return P

def sim_for_counterexample(system_params, rho=0):
    M, F, T, Vr, x, M_func, f_func, q, u, x0 = system_params
    states = np.array(q + u)
    try:
        samples = np.load(folder + '/counterexample-x-sample-mesh.npy')
        V = np.load(folder + '/counterexample-V-evals.npy')
    except:
        aa = np.linspace(-.5, .5, 1)
        bb = np.meshgrid(*([aa] * (2 * n + 2)))
        samples = np.array([i.ravel() for i in bb]).T
        samples = np.array([i + x0 for i in samples])
        V = [Vr.subs(dict(zip(x, original_to_x(i)))) for i in samples]
        # do sorting
        samples = samples[np.argsort(V)]
        V = np.array(np.sort(V), dtype=float)
        np.save(folder + '/counterexample-x-sample-mesh.npy', samples)
        np.save(folder + '/counterexample-V-evals.npy', V)

    int_samples = samples[V >= rho]
    print(int_samples[0])
    for i in int_samples:
        sol = integrate.solve_ivp(right_hand_side, [0, 12], i, events=event)
        if sol.status == 1:
            print(i)
            print(sol.y[:, -1])
            print(Vr.subs(dict(zip(x, original_to_x(i)))))
            return None


def event(t, x):
    x[1:n + 1] = x[1:n + 1] % (2 * np.pi)
    norm = np.linalg.norm(x - x0)
    out_ = norm - 20
    in_ = norm - 1e-2
    return out_ or in_
event.terminal = True


n = 3
print(n)
folder = './link' + str(n)

system_params = NlinkedSys(n)
M, F, T, Vr, x, M_func, f_func, q, u, x0 = system_params
# sim_for_counterexample(system_params, rho=.36)
# alternative_fixedpt(system_params)
# sample_parameterization(system_params, 10)
sample(system_params, 500)
# get_Y(Vr, x)

# Y = load_x_and_y()
# [xx, xxd, psi, V] = Y
# print(xx.shape[0])
# print(min(V))
#
# transformed_basis, T = coordinate_ring_transform(psi, True, False)
# print(check_genericity(psi, False))
# Y[2] = transformed_basis
# rho, P, Y = solve_SDP_on_samples(Y, write_to_file=False)

# x0 = x_to_originial(xx[5])
# do_anim(x0)
