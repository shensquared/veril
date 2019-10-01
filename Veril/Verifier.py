import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from scipy.linalg import solve_discrete_lyapunov
import pydrake
import numpy as np
from numpy.linalg import eig
import pydrake.symbolic as sym
from pydrake.all import (MathematicalProgram, Polynomial, SolutionResult,
   Solve, Jacobian)
import Plants
from keras import backend as K

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')





def get_P0(CL):
    plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    prog = MathematicalProgram()
    # add the constant basis
    # constant = prog.NewIndeterminates(1, 'constant')
    # basis = [sym.Monomial(constant[0], 0)]
    x = prog.NewIndeterminates(plant.num_states, "x")
    c = prog.NewIndeterminates(CL.units, "c")
    full_states = np.hstack((x, c))
    basis = [sym.Monomial(_) for _ in full_states]
    full_dim = plant.num_states + CL.units

    A0 = CL.linearize()
    P = prog.NewSymmetricContinuousVariables(full_dim, "P")
    prog.AddPositiveSemidefiniteConstraint(P)
    prog.AddPositiveSemidefiniteConstraint(P + P.T)
    V = full_states.T@P@full_states
    Vdot = full_states.T@P@A0@full_states+full_states.T@A0.T@P@full_states
    if plant.manifold is not None:
        r = prog.NewContinuousVariables(1, "r")[0]
        Vdot = Vdot + r * plant.manifold(x)
    slack = prog.NewContinuousVariables(1, "s")[0]

    prog.AddConstraint(slack >= 0)
    prog.AddSosConstraint(-Vdot - slack * full_states.T@np.eye
        (full_dim)@full_states)
    prog.AddCost(-slack)
    result = Solve(prog)
    print('w/ solver %s' % (result.get_solver_id().name()))
    print(result.get_solution_result())
    P = result.GetSolution(P)
    print(eig(A0)[0])
    print(eig(A0.T@P+P@A0)[0])

#
# class SOS_verifier():
#
#     def __init__(self, tag, num_inputs=1, num_nuerons=10, zero_nominal=False,
#                  step_or_total='step'):
#         self.zero_nominal = zero_nominal
#         self.num_inputs = num_inputs
#         self.num_nuerons = num_nuerons
#
#         else:
#             sys.exit(1)
#
#         # self.aug_dyanmics = np.array([.5, 0, .5, .5, -.5])
#         self.aug_dyanmics = np.kron(np.eye(self.num_nuerons),
#                                     np.array([.5, 0, .5, .5, -.5]))
#         self.setup_prog()
#
#     def setup_prog(self):
#         prog = MathematicalProgram()
#         # add the constant basis
#         constant = prog.NewIndeterminates(1, 'constant')
#         basis = [sym.Monomial(constant[0], 0)]
#
#         xhat = prog.NewIndeterminates(3 * self.num_nuerons, "xhat")
#         uhat = prog.NewIndeterminates(self.num_inputs, "uhat")
#         basis += [sym.Monomial(_) for _ in np.hstack((uhat, xhat))]
#         # set up the nominal state and inputs
#         if self.zero_nominal:
#             u = np.zeros((self.num_inputs,))
#             x = np.zeros((3 * self.num_nuerons))
#         else:
#             x = prog.NewIndeterminates(3 * self.num_nuerons, "x")
#             u = prog.NewIndeterminates(self.num_inputs, "u")
#             basis += [sym.Monomial(_) for _ in np.hstack((x, u))]
#
#         self.prog = prog
#         self.xhat = xhat
#         self.uhat = uhat
#         self.u = u
#         self.x = x
#         self.basis = basis
#
#         self.x1 = x.reshape(self.num_nuerons, 3)[:, 0]
#         self.xf = x.reshape(self.num_nuerons, 3)[:, 1]
#         self.xc = x.reshape(self.num_nuerons, 3)[:, 2]
#
#         # Set up the actual states and inputs
#         self.xhat1 = xhat.reshape(self.num_nuerons, 3)[:, 0]
#         self.xhatf = xhat.reshape(self.num_nuerons, 3)[:, 1]
#         self.xhatc = xhat.reshape(self.num_nuerons, 3)[:, 2]
#
#         # calculate the output
#         self.dx = xhat - x
#         self.du = uhat - u
#
#         self.dx1 = self.dx.reshape(self.num_nuerons, 3)[:, 0]
#         self.dy = self.dx1.dot(self.output_weight)
#
#         # calculate the error dynamics
#         self.x1_plus = self.poly_dynamics(x)
#         self.xhat1_plus = self.poly_dynamics(xhat)
#         self.dx1_plus = self.xhat1_plus - self.x1_plus
#
#     def get_args(self, state, inputs):
#         x_f = np.dot(inputs, self.kernel_f)
#         x_c = np.dot(inputs, self.kernel_c)
#         x_f = np.add(x_f, self.bias_f)
#         x_c = np.add(x_c, self.bias_c)
#         arg_f = (x_f + np.dot(state, self.recurrent_kernel_f))
#         arg_c = (x_c + np.dot(state, self.recurrent_kernel_c))
#         return arg_f, arg_c
#
#     def explicit_Jacobian(self):
#         c = self.xhat1
#         u = self.uhat
#         arg_f, arg_c = self.get_args(c, u)
#         tau_f = np.array([sym.tanh(_) for _ in arg_f])
#         tau_c = np.array([sym.tanh(_) for _ in arg_c])
#         # split into 4 parts, so to get around drake Jacobian issues
#         Jac1 = Jacobian(c, c)
#         Jac2 = (Jacobian(tau_f, c).T * c).T + (Jacobian(c, c).T * tau_f).T
#         Jac3 = Jacobian(tau_c, c)
#         Jac4 = -(Jacobian(tau_c, c).T * tau_f).T - (Jacobian(tau_f, c).T *
#                                                     tau_c).T
#         Jac = .5 * (Jac1 + Jac2 + Jac3 + Jac4)
#         # evaluate at zero
#         env = {_: 0 for _ in np.hstack((c, u))}
#         self.linerized_A = np.array(
#             [[x.Evaluate(env) for x in J] for J in Jac])
#         self.linerized_S0 = solve_discrete_lyapunov(
#             self.linerized_A.T, np.eye(self.num_nuerons))
#         print eig(self.linerized_S0)[0]
#
#     def test_explicit_linearize(self):
#         prog = MathematicalProgram()
#         c = prog.NewIndeterminates(self.num_nuerons, "c")
#         u = prog.NewIndeterminates(1, "u")
#         # random and non-zero sample
#         env = {_: np.random.uniform(5, 10, 1) for _ in np.hstack((c, u))}
#         arg_f, arg_c = self.get_args(c, u)
#         tau_f = np.array([sym.tanh(_) for _ in arg_f])
#         tau_c = np.array([sym.tanh(_) for _ in arg_c])
#         # direct jacobian method
#         c_plus1 = ((c)).tolist()
#         c_plus2 = ((tau_f * c)).tolist()
#         c_plus3 = ((tau_c)).tolist()
#         c_plus4 = ((- tau_c * tau_f)).tolist()
#         direct_Jac1 = np.array([[x.Evaluate(env) for x in J]
#                                 for J in Jacobian(c_plus1, c)])
#         direct_Jac2 = np.array([[x.Evaluate(env) for x in J]
#                                 for J in Jacobian(c_plus2, c)])
#         direct_Jac3 = np.array([[x.Evaluate(env) for x in J]
#                                 for J in Jacobian(c_plus3, c)])
#         direct_Jac4 = np.array([[x.Evaluate(env) for x in J]
#                                 for J in Jacobian(c_plus4, c)])
#         directJac = .5 * (direct_Jac1 + direct_Jac2 +
#                           direct_Jac3 + direct_Jac4)
#         # explicit jacobian
#         Jac1 = np.array(Jacobian(c, c))
#         Jac2 = np.array((Jacobian(tau_f, c).T * c).T +
#                         (Jacobian(c, c).T * tau_f).T)
#         Jac3 = np.array(Jacobian(tau_c, c))
#         Jac4 = np.array(-(Jacobian(tau_c, c).T * tau_f).T - (Jacobian(tau_f, c).T *
#                                                              tau_c).T)
#         Jac = .5 * (Jac1 + Jac2 + Jac3 + Jac4)
#         manualJac = np.array([[x.Evaluate(env) for x in J] for J in Jac])
#         print manualJac - directJac
#
#     def poly_dynamics(self, states):
#         x1 = states.reshape(self.num_nuerons, 3)[:, 0]
#         xf = states.reshape(self.num_nuerons, 3)[:, 1]
#         xc = states.reshape(self.num_nuerons, 3)[:, 2]
#         return .5 * (x1 + x1 * xf - xf * xc + xc)
#
#     def bilinear(self, converged_tol=.01, degL1=2, degL2=2, degV=2, max_iterations=10):
#         T, S0bal = balance(self.linerized_S0, self.linerized_A)
#         rho = 1
#         vol = 0
#         for iter in range(max_iterations):
#             last_vol = vol
#
#             # % balance on every iteration (since V and Vdot are changing):
#             T, Sbal = balance(self.linerized_S0 / rho, self.linerized_A)
#             S0bal = (T.T).dot(self.linerized_S0).dot(T)
#
#             # do the alternation
#             r, l, m, n, sigma1 = self.find_n(Sbal)
#             scales = find_scales(Sbal, S0bal, rho)
#             Sbal, r, l, m, rho = optimizeV(n, scales, S0bal, sigma1)
#             vol = rho
#
#             # % undo balancing (for the next iteration, or if i'm done)
#             T_inv = np.linalg.inv(T)
#             S = (T_inv.T).dot(S).dot(T_inv)
#
#             # % check for convergence
#             if (vol - last_vol) < (converged_tol * last_vol):
#                 break
#         return S
#
#     def find_n(self, S=None):
#         prog = self.prog
#
#         # if S is None:
#         #     S = prog.NewSymmetricContinuousVariables(self.num_nuerons, "S")
#         #     prog.AddPositiveSemidefiniteConstraint(S)
#         #     prog.AddPositiveSemidefiniteConstraint(S + S)
#
#         V = self.xhat1.dot(S).dot(self.xhat1)
#         Vplus = self.xhat1_plus.dot(S).dot(self.xhat1_plus)
#         # V = prog.NewFreePolynomial(sym.Variables(xhat1), 2)
#         # V = prog.NewSosPolynomial(sym.Variables(xhat1), 2)
#
#         # Vplus = prog.SubstituteSolution(sym.Polynomial(V, {sym.Monomial(xhat1[0]):dx1_plus[0]}))
#         # {sym.Monomial(xhat1[0]):dx1_plus[0]}
#         # poly_sub_expected = sym.Polynomial(
#         # prog.SubstituteSolution(V, dx1_plus[0]))
#
#         # Vplus = dx1_plus.T.dot(S).dot(dx1_plus)
#         dV = Vplus - V
#
#         # the storage and the storage plus
#         # poly = prog.NewFreePolynomial(sym.Variables([x1, xhat1]), 2)
#         # env = dict(zip(x1, x1_plus))
#         # env_new = dict(zip(xhat1, xhat1_plus))
#         # env.update(env_new)
#         # # V = dx1.T.dot(S).dot(dx1)
#         # poly_plus = poly.Evaluate(env)
#         # dV = poly_plus - poly
#
#         # supply rate:
#         # r = prog.NewContinuousVariables(1, "r")[0]
#         # prog.AddConstraint(r >= 0)
#         # supply_rate = (self.dy.T.dot(self.dy) - (r * self.du.dot(self.du)))
#
#         # dissipation = dV + supply_rate
#
#         # the equality constraints
#         # m = prog.NewContinuousVariables(4 * self.num_nuerons, "m")
#         # eqltyCstr = np.hstack((x4 - x1 * xf,
#         #                        x5 - xc * xf,
#         #                        xhat4 - xhat1 * xhatf,
#         #                        # sym.pow((xhat4 - xhat1 * xhatf)[0],2),
#         #                        xhat5 - xhatc * xhatf,
#         #                        # sym.pow((xhat5 - xhatc * xhatf)[0],2)
#         #                        )).dot(m)
#
#         # the IQCs
#         # l = []
#         # for i in range(4 * 4 * self.num_nuerons):
#         #     if self.zero_nominal:
#         #         (_, binding) = prog.NewSosPolynomial(sym.Variables(xhat), 2)
#         #         l.append(_.ToExpression())
#         #     else:
#         #         (_, binding) = prog.NewSosPolynomial(
#         #             (sym.Monomial(xhat, 2), sym.Monomial(x, 2)))
#         #         l.append(_.ToExpression())
#
#         l = prog.NewContinuousVariables(4 * 4 * self.num_nuerons, "l")
#         for i in l:
#             prog.AddConstraint(i >= 0)
#
#         arg_f, arg_c = self.get_args(self.x1, self.u)
#         arg_fhat, arg_chat = self.get_args(self.xhat1, self.uhat)
#
#         IQC_Cstr = np.hstack((
#             IQC_tanh(arg_f, self.xf),
#             IQC_tanh(arg_c, self.xc),
#             IQC_tanh(arg_fhat, self.xhatf),
#             IQC_tanh(arg_chat, self.xhatc)
#         ))
#         IQC_Cstr = IQC_Cstr.dot(l)
#
#         # poly = sym.Polynomial(self.poly_dynamics(
#         #     xhat, u)[0] * self.poly_dynamics(xhat, u)[0])
#         # print poly.ConstructMonomialBasis()
#
#         # poly=sym.Polynomial(self.poly_dynamics(xhat,u)[0])
#         # print poly.ConstructMonomialBasis()
#
#         n = prog.NewContinuousVariables(2, "n")
#         prog.AddConstraint(n[0] >= 0)
#         small_deviation = n[0] * (V - 1)
#         # prog.AddConstraint(n[1] >= 0)
#         # small_disturb = n[1] * (self.du.dot(self.du) - 1e-8)
#
#         sigma1 = prog.NewContinuousVariables(1, 'sigma1')
#         prog.AddConstraint(sigma1[0] >= 0)
#
#         cstr = - dV + IQC_Cstr + small_deviation - sigma1 * V
#
#         prog.AddSosConstraint(cstr[0])
#         prog.AddCost(-sigma1[0])
#         result = prog.Solve()
#         print result
#
#         if result == SolutionResult.kSolutionFound:
#             r = prog.GetSolution(r)
#             l = prog.GetSolution(l)
#             m = prog.GetSolution(m)
#             n = prog.GetSolution(n)
#             return r, l, m, n, sigma1
#         else:
#             print result
#             raise RuntimeError(result)
#     # def verify_l2(self):
#     #     prog = MathematicalProgram()
#     #     # add the constant basis
#     #     cons = prog.NewIndeterminates(1, 'cons')
#     #     basis = [sym.Monomial(cons[0], 0)]
#
#     #     xhat = prog.NewIndeterminates(5 * self.num_nuerons, "xhat")
#     #     basis += [sym.Monomial(_) for _ in xhat]
#     #     uhat = prog.NewIndeterminates(1, "uhat")
#     #     basis += [sym.Monomial(_) for _ in uhat]
#
#     #     # set up the nominal state and inputs
#     #     if self.zero_nominal:
#     #         u = np.zeros((1,))
#     #         x = np.zeros((5 * self.num_nuerons))
#     #     else:
#     #         x = prog.NewIndeterminates(5 * self.num_nuerons, "x")
#     #         u = prog.NewIndeterminates(1, "u")
#     #         basis += [sym.Monomial(_) for _ in x]
#     #         basis += [sym.Monomial(_) for _ in u]
#     #     self.basis = basis
#     #     x1 = x.reshape(self.num_nuerons, 5)[:, 0]
#     #     xf = x.reshape(self.num_nuerons, 5)[:, 1]
#     #     xc = x.reshape(self.num_nuerons, 5)[:, 2]
#     #     x4 = x.reshape(self.num_nuerons, 5)[:, 3]
#     #     x5 = x.reshape(self.num_nuerons, 5)[:, 4]
#     #     # Set up the actual states and inputs
#     #     xhat1 = xhat.reshape(self.num_nuerons, 5)[:, 0]
#     #     xhatf = xhat.reshape(self.num_nuerons, 5)[:, 1]
#     #     xhatc = xhat.reshape(self.num_nuerons, 5)[:, 2]
#     #     xhat4 = xhat.reshape(self.num_nuerons, 5)[:, 3]
#     #     xhat5 = xhat.reshape(self.num_nuerons, 5)[:, 4]
#
#     #     # calculate the output
#     #     dx = xhat - x
#     #     du = uhat - u
#     #     dx1 = dx.reshape(self.num_nuerons, 5)[:, 0]
#     #     dy = dx1.dot(self.output_weight)
#     #     # calculate the error dynamics
#     #     dx1_plus = self.aug_dyanmics.dot(dx)
#
#     #     # Lyapunov_like step decrease:
#     #     S = prog.NewSymmetricContinuousVariables(self.num_nuerons, "S")
#     #     # prog.AddPositiveSemidefiniteConstraint(S)
#     #     prog.AddPositiveSemidefiniteConstraint(
#     #         S - 1e-10 * np.eye(self.num_nuerons))
#
#     #     V = dx1.T.dot(S).dot(dx1)
#     #     Vplus = dx1_plus.T.dot(S).dot(dx1_plus)
#     #     dV = Vplus - V
#
#     #     # supply rate:
#     #     r = prog.NewContinuousVariables(1, "r")
#     #     prog.AddConstraint(r >= 0)
#     #     supply_rate = (r * dy.T.dot(dy) - du.dot(du))
#     #     dissipation = dV + supply_rate
#
#     #     # the equality constraints
#     #     m = prog.NewContinuousVariables(4 * self.num_nuerons, "m")
#     #     eqltyCstr = np.hstack((x4 - x1 * xf,
#     #                            x5 - xc * xf,
#     #                            xhat4 - xhat1 * xhatf,
#     #                            xhat5 - xhatc * xhatf
#     #                            )).dot(m)
#
#     #     # the IQCs
#     #     l = prog.NewContinuousVariables(4 * 4 * self.num_nuerons, "l")
#     #     for i in l:
#     #         prog.AddConstraint(i >= 0)
#
#     #     arg_f, arg_c = self.get_args(x1, u)
#     #     arg_fhat, arg_chat = self.get_args(xhat1, uhat)
#
#     #     IQC_Cstr = np.hstack((
#     #         IQC_tanh(arg_f, xf),
#     #         IQC_tanh(arg_c, xc),
#     #         IQC_tanh(arg_fhat, xhatf),
#     #         IQC_tanh(arg_chat, xhatc)
#     #     )).dot(l)
#
#     #     if self.step_or_total is 'step':
#     #         # |du| less than 1e-3
#     #         a = prog.NewContinuousVariables(1, "a")
#     #         prog.AddConstraint(a[0] >= 0)
#     #         small_disturb = a * (du.T.dot(du) - 1e-8)
#     #         small_deviation = 0
#     #     elif self.step_or_total is 'total':
#     #         # |dx| less than 1e-3
#     #         b = prog.NewContinuousVariables(1, "b")
#     #         prog.AddConstraint(b[0] >= 0)
#     #         small_deviation = b * (dx1.T.dot(dx1) - 1e-6)
#     #         small_disturb = 0
#     #     elif self.step_or_total is 'both':
#     #         a = prog.NewContinuousVariables(1, "a")
#     #         prog.AddConstraint(a[0] >= 0)
#     #         small_disturb = a * (du.T.dot(du) - 1e-10)
#     #         b = prog.NewContinuousVariables(1, "b")
#     #         prog.AddConstraint(b[0] >= 0)
#     #         small_deviation = b * (dx1.T.dot(dx1) - 1e-10)
#
#     #     prog.AddSosConstraint((- dissipation +
#     #                            eqltyCstr +
#     #                            IQC_Cstr +
#     #                            small_disturb +
#     #                            small_deviation)[0],
#     #                           basis)
#     #     prog.AddCost(-r)
#     #     result = prog.Solve()
#
#     #     if result == SolutionResult.kSolutionFound:
#     #         r = prog.GetSolution(r)
#     #         print r
#     #         S = prog.GetSolution(S)
#     #         # print eig(S)[0]
#     #         return r
#     #     else:
#     #         print result
#     #         return -1
#
#
# def find_scales(S, S0, rho):
#     prog = MathematicalProgram()
#     prog = prog.withIndeterminate(x)
#
#     [prog, L2] = prog.newFreePoly(Lxmonom)
#
#     [prog, slack] = prog.newPos(1)
#
#     prog = prog.withSOS(-(V - 1) + L2 * (V0 - rho))
#     prog = prog.withSOS(L2)
#
#     prog.AddCost(-slack[0])
#     result = prog.Solve()
#
#     if result == SolutionResult.kSolutionFound:
#         scales = prog.GetSolution(scales)
#         return scales
#     else:
#         print result
#         raise RuntimeError(result)
#
#
# def optimizeV(L1, L2, V0, sigma1):
#     prog = spotsosprog
#     prog = prog.withIndeterminate(x)
#
#     [prog, V] = prog.newFreePoly(Vxmonom)
#     Vdot = diff(V, x) * f
#
#     [prog, rho] = prog.newPos(1)
#
#     prog = prog.withSOS(-Vdot + L1 * (V - 1) - sigma1 * V / 2)
#     prog = prog.withSOS(-(V - 1) + L2 * (V0 - rho))
#     prog = prog.withSOS(V)
#
#     prog.AddCost(-rho[0])
#     result = prog.Solve()
#
#     if result == SolutionResult.kSolutionFound:
#         S = prog.GetSolution(S)
#         r = prog.GetSolution(r)
#         l = prog.GetSolution(l)
#         m = prog.GetSolution(m)
#         rho = prog.GetSolution(rho)
#         return S, r, l, m, rho
#     else:
#         print result
#         raise RuntimeError(result)
#
#
# def IQC_tanh(x, y):
#     y_cross = 0.642614
#     x_off = .12
#     return np.hstack((y ** 2 - 1,
#                       (y - ((x) + x_off)) * (y + y_cross),
#                       (y - ((x) - x_off)) * (y - y_cross),
#                       (y - x) * y))
#
#
# def balanceQuadraticForm(S, P):
#     # copied from the old drake, with only syntax swap
#     #  Quadratic Form "Balancing"
#     #
#     #    T = balqf(S,P)
#     #
#     #  Input:
#     #    S -- n-by-n symmetric positive definite.
#     #    P -- n-by-n symmetric, full rank.
#     #
#     #  Finds a T such that:
#     #    T'*S*T = D
#     #    T'*P*T = D^(-1)
#
#     # if np.linalg.norm(S - S.T,1) > 1e-8:
#     # raise Error('S must be symmetric')
#     # if np.linalg.norm(P - P.T,1) > 1e-8:
#     # raise Error('P must be symmetric')
#     # if np.linalg.cond(P) > 1e10:
#     # raise Error('P must be full rank')
#
#     # Tests if S positive def. for us.
#     V = np.linalg.inv(np.linalg.cholesky(S).T)
#     [U, l, N] = np.linalg.svd((V.T.dot(P)).dot(V))
#     T = (V.dot(U)).dot(np.diag(np.power(l, -.25, dtype=float)))
#     D = np.diag(np.power(l, -.5, dtype=float))
#     return T, D
#
#
# def balance(S, A, x=None, f=None):
#     # [T, D] = balanceQuadraticForm(S, (S.dot(A) + A.T.dot(S)))
#     [T, D] = balanceQuadraticForm(S, (A.T.dot(S).dot(A) - S))
#     Sbal = (T.T).dot(S).dot(T)
#     if f is not None:
#         fbal = inv(T) * subs(f, x, T * x)
#         return T, Sbal, fbal
#     return T, Sbal
