import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")

import pydrake
import numpy as np
from numpy.linalg import eig, inv
import pydrake.symbolic as sym
from pydrake.all import (MathematicalProgram, Polynomial, SolutionResult,
                         Solve, Jacobian, Evaluate, RealContinuousLyapunovEquation, Substitute)
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
from Veril.Verifier import *


# plant = 'Cubic'
plant = 'vdp'

prog = MathematicalProgram()
if plant is 'Cubic':
    nx = 1
    x = prog.NewIndeterminates(nx, "x")
    xdot = [-x[0] + x[0]**3]
elif plant is 'vdp':
    nx=2
    x = prog.NewIndeterminates(nx, "x")
    xdot = -np.array([x[1], -x[0] - x[1] * (x[0]**2 - 1)])

options = opt(nx, converged_tol = 1e-2, degL1=6, degL2=6, degV=4,
   max_iterations=100)
J = Jacobian(xdot, x)
env = dict(zip(x, np.zeros(x.shape)))
A = np.array([[i.Evaluate(env) for i in j]for j in J])
S0 = solve_lyapunov(A.T, -np.eye(nx))
V0 = (x.T@S0@x)

bilinear(V0, xdot, S0, A, options)


def levelsetMethod(x, f, V, options):
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    # % construct multipliers for Vdot
    L1 = prog.NewFreePolynomial(Variables(x), options.degL1).ToExpression()
    # % construct Vdot
    Vdot = clean(V.Jacobian(x) @ f)
    # % construct slack var
    sigma1 = prog.NewContinuousVariables(1, "s")[0]
    prog.AddSosConstraint((x.T@x)**4 * (V - sigma1) + L1 * Vdot)
    # add cost
    prog.AddCost(-sigma1)

    solver = MosekSolver()
    solver.set_stream_logging(False, "")
    result = solver.Solve(prog, None, None)
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    L1 = result.GetSolution(L1)
    sigma1 = result.GetSolution(sigma1)
    print(sigma1)
    return sigma1


# def findL1(old_x, f, V, options):
#     prog = MathematicalProgram()
#     x = prog.NewIndeterminates(options.nX,'lx')
#     V = V.Substitute(dict(zip(old_x, x)))
#     f = [i.Substitute(dict(zip(old_x, x))) for i in f]
#     # % construct multipliers for Vdot
#     L1 = prog.NewSosPolynomial(Variables(x), options.degL1)
#     L1 = L1[0].ToExpression()
#     print(L1)
#     # % construct Vdot
#     # x = list(V.GetVariables())
#     Vdot = (V.Jacobian(x) @ f)
#     # print('Vdot')
#     print(Vdot)
#     # % construct slack var
#     sigma1 = prog.NewContinuousVariables(1, "s")[0]
#     prog.AddConstraint(sigma1 >= 0)
#     # % setup SOS constraints
#     # print(Polynomial(-Vdot + L1 * (V - 1) - sigma1 * V).TotalDegree())
#     prog.AddSosConstraint(-Vdot + L1 * (V - 1) - sigma1 * V)
#     # add cost
#     prog.AddCost(-sigma1)
#     result = Solve(prog)
#     print('w/ solver %s' % (result.get_solver_id().name()))
#     print(result.get_solution_result())
#     assert result.is_success()
#     L1 = result.GetSolution(L1)
#     sigma1 = result.GetSolution(sigma1)
#     print(sigma1)
#     return L1, sigma1

# findL1(x, xdot, V0, options)
