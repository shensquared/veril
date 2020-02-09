from __future__ import division, print_function
import sympy as sm
from sympy import Matrix
import sympy.physics.mechanics as me
me.init_vprinting()

import numpy as np
from numpy.linalg import solve
# from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
# Now specify the number of links, $n$. I'll start with 5 since the Wolfram folks only showed four.

n = 4
# loaded  = pickle.load(open(str(n) + ".p", "rb" ))

q = me.dynamicsymbols('q:{}'.format(n + 1))  # Generalized coordinates
u = me.dynamicsymbols('u:{}'.format(n + 1))  # Generalized speeds
f = me.dynamicsymbols('f')                   # Force applied to the cart

# m = sm.symbols('m:{}'.format(n + 1))         # Mass of each bob
m = np.ones(n+1,)
# l = sm.symbols('l:{}'.format(n))             # Length of each link
l = .5*np.ones(n+1,)
g =9.81
t = sm.symbols('t')                     # Gravity and time

I = me.ReferenceFrame('I')  # Inertial reference frame
O = me.Point('O')           # Origin point
O.set_vel(I, 0)             # Origin's velocity is zero

# Secondly, we define the define the first point of the pendulum as a particle which has mass. This point can only move laterally and represents the motion of the "cart".

P0 = me.Point('P0')                 # Hinge point of top link
P0.set_pos(O, q[0] * I.x)           # Set the position of P0
P0.set_vel(I, u[0] * I.x)           # Set the velocity of P0
Pa0 = me.Particle('Pa0', P0, m[0])  # Define a particle at P0

# Now we can define the $n$ reference frames, particles, gravitational forces, and kinematical differential equations for each of the pendulum links. This is easily done with a loop.
frames = [I]                              # List to hold the n + 1 frames
points = [P0]                             # List to hold the n + 1 points
particles = [Pa0]                         # List to hold the n + 1 particles
forces = [(P0, f * I.x - m[0] * g * I.y)] # List to hold the n + 1 applied forces, including the input force, f
kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's

for i in range(n):
    Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])   # Create a new frame
    Bi.set_ang_vel(I, u[i + 1] * I.z)                         # Set angular velocity
    frames.append(Bi)                                         # Add it to the frames list

    Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)  # Create a new point
    Pi.v2pt_theory(points[-1], I, Bi)                         # Set the velocity
    points.append(Pi)                                         # Add it to the points list

    Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])        # Create a new particle
    particles.append(Pai)                                     # Add it to the particles list

    forces.append((Pi, -m[i + 1] * g * I.y))                  # Set the force applied at the point

    kindiffs.append(q[i + 1].diff(t) - u[i + 1])              # Define the kinematic ODE:  dq_i / dt - u_i = 0


# With all of the necessary point velocities and particle masses defined, the `KanesMethod` class can be used to derive the equations of motion of the system automatically.

kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)  # Initialize the object
fr, frstar = kane.kanes_equations(forces, particles)         # Generate EoM's fr + frstar = 0

states = q + u    # Make a list of the states
# Controller Design
# =================
equilibrium_point = [sm.S(0)] + [sm.pi / 2] * (len(q) - 1) + [sm.S(0)] * len(u)
equilibrium_dict = dict(zip(q + u, equilibrium_point))

# The `KanesMethod` class has method that linearizes the forcing vector about generic state and input perturbation vectors. The equilibrium point and numerical constants can then be substituted in to give the linear system in this form: $M\dot{x}=F_Ax+F_Br$. The state and input matrices, $A$ and $B$, can then be computed by left side multiplication by the inverse of the mass matrix: $A=M^{-1}F_A$ and $B=M^{-1}F_B$.

M, F_A, F_B, r = kane.linearize(new_method=True, op_point=equilibrium_dict)

M_num = sm.matrix2numpy(M, dtype=float)
F_A_num = sm.matrix2numpy(F_A, dtype=float)
F_B_num = sm.matrix2numpy(F_B, dtype=float)
A = np.linalg.solve(M_num, F_A_num)
B = np.linalg.solve(M_num ,F_B_num)

# Also convert `equilibrium_point` to a numeric array:
equilibrium_point = np.asarray([x.evalf() for x in equilibrium_point], dtype=float)

# So now we can compute the optimal gains with a linear quadratic regulator. I chose identity matrices for the weightings for simplicity.
Q = np.eye(A.shape[0])
R = np.eye(B.shape[1])
S = solve_continuous_are(A, B, Q, R);
K = np.dot(np.dot(np.linalg.inv(R), B.T),  S)

# The gains can now be used to define the required input during simulation to stabilize the system. The input $r$ is simply the gain vector multiplied by the error in the state vector from the equilibrium point, $r(t)=K(x_{eq} - x(t))$.

# def right_hand_side(x, t, args):
#     """Returns the derivatives of the states.

#     Parameters
#     ----------
#     x : ndarray, shape(2 * (n + 1))
#         The current state vector.
#     t : float
#         The current time.
#     args : ndarray
#         The constants.

#     Returns
#     -------
#     dx : ndarray, shape(2 * (n + 1))
#         The derivative of the state.

#     """
#     r = np.dot(K, equilibrium_point - x)    # The controller
#     arguments = np.hstack((x, r, args))     # States, input, and parameters
#     dx = np.array(solve(M_func(*arguments), # Solving for the derivatives
#                         f_func(*arguments))).T[0]

#     return dx


# Now we can simulate and animate the system to see if the controller works.

# x0 = np.hstack((0,
#                 np.pi / 2 * np.ones(len(q) - 1),
#                 1 * np.ones(len(u))))
# t = np.linspace(0.0, 10.0, num=500)
# x = odeint(right_hand_side, x0, t, args=(parameter_vals,))

#################
forces = [(P0, np.dot(K, equilibrium_point - states)[0] * I.x - m[0] * g *
 I.y)] # List to hold the n + 1 applied forces, including the input force, f
for i in range(n):
    forces.append((Pi, -m[i + 1] * g * I.y))                  # Set the force applied at the point

kane_cl = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
fr_cl, frstar_cl = kane_cl.kanes_equations(forces, particles)
# M_cl, F_A_cl, F_B_cl, r = kane_cl.linearize(new_method=True,
#    op_point=equilibrium_dict)

massmatrix = sm.trigsimp(kane_cl.mass_matrix)
forcesmatrix = sm.trigsimp(kane_cl.forcing)
print(massmatrix)
print(forcesmatrix)
dyanmics = sm.simplify(massmatrix.inv()@forcesmatrix)
[print(i) for i in dyanmics]
# import pickle
# pickle.dump(dyanmics, open(str(n) + ".p", "wb"))
#
# from sympy import series
# exp = series(q[0]**4,x = tuple(q), n=3)
#
# aaa = sm.simplify(dyanmics[0]).series(x= tuple(states),x0=equilibrium_point,n=3)
# print(aaa)
# # velocity = np.array(u).reshape((n+1,1))
# dyanmics = dyanmics.row_insert(0, Matrix(u))
# # print(dyanmics)
#
# first = dyanmics.jacobian(states)
# second = first.jacobian(states)
# third = second.jacobian(states)
#
# first_num = sm.matrix2numpy(first.subs(equilibrium_dict), dtype=float)
# second_num = sm.matrix2numpy(second.subs(equilibrium_dict), dtype=float)
# third_num = sm.matrix2numpy(third.subs(equilibrium_dict), dtype=float)
#
# print(first)

