import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (MathematicalProgram, Polynomial,
                         Expression, SolutionResult,
                         Variables, Solve, Jacobian, Evaluate,
                         RealContinuousLyapunovEquation, Substitute,
                         MosekSolver, MonomialBasis)


class opt:

    def __init__(self, nX, degf, degV=4, converged_tol=.01, max_iterations=10,
                 degL1=None, degL2=None, do_balance=False):
        self.nX = nX
        self.degf = degf
        self.degV = degV
        self.converged_tol = converged_tol
        self.max_iterations = max_iterations
        if degL1 is None:
            degL1 = degV - 1 + degf
        self.degL1 = degL1
        if degL2 is None:
            degL2 = degL1
        self.degL2 = degL2
        self.do_balance = do_balance


def bilinear(x, V0, f, S0, A, options):
    # x = np.array(list(V0.GetVariables()))
    V = V0
    if options.do_balance:
        [T, V0bal, fbal, S0, A] = balance(x, V0, f, S0, A)
    else:
        T, V0bal, fbal = np.eye(options.nX), V0, f
    rho = 1
    vol = 0
    for iter in range(options.max_iterations):
        # print('iteration  %s' % (iter))
        last_vol = vol

        # balance on every iteration (since V and Vdot are changing):
        if options.do_balance:
            [T, Vbal, fbal] = balance(x, V, f, S0 / rho, A)[0:3]
        else:
            T, Vbal, fbal = np.eye(options.nX), V, f
        # print('T is %s' % (T))
        V0bal = V0.Substitute(dict(zip(x, T@x)))
        # env = dict(zip(list(V0bal.GetVariables()), np.array([1, 2.31])))
        # print('V0bal is %s' % (V0bal.Evaluate(env)))

        [L1, sigma1] = findL1(x, fbal, Vbal, options)
        L2 = findL2(x, Vbal, V0bal, rho, options)
        [Vbal, rho] = optimizeV(x, fbal, L1, L2, V0bal, sigma1, options)
        vol = rho

        #  undo balancing (for the next iteration, or if i'm done)
        V = Vbal.Substitute(dict(zip(x, inv(T)@x)))
        if ((vol - last_vol) < options.converged_tol * last_vol):
            break
    # print('final rho is %s' % (rho))
    # env = dict(zip(x, np.array([1, 2.31])))
    # print('V is %s' % (V.Evaluate(env)))
    return V


def findL1(x, f, V, options):
    # print('finding L1')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    # % construct multipliers for Vdot
    L1 = prog.NewFreePolynomial(Variables(x), options.degL1).ToExpression()

    # % construct Vdot
    Vdot = clean(V.Jacobian(x) @ f)
    # env = dict(zip(x, np.array([1, 2.31])))
    # print('f0 is %s' % (f[0].Evaluate(env)))
    # print('f1 is %s' % (f[1].Evaluate(env)))
    # print('V is %s' % (V.Evaluate(env)))
    # % construct slack var
    sigma1 = prog.NewContinuousVariables(1, "s")[0]
    prog.AddConstraint(sigma1 >= 0)
    # % setup SOS constraints
    prog.AddSosConstraint(-Vdot + L1 * (V - 1) - sigma1 * V)
    prog.AddSosConstraint(L1)
    # add cost
    prog.AddCost(-sigma1)
    # result = Solve(prog)
    solver = MosekSolver()
    solver.set_stream_logging(True, "")
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    assert result.is_success()
    L1 = (result.GetSolution(L1))
    sigma1 = result.GetSolution(sigma1)
    # print('sigma1 is %s' % (sigma1))
    return L1, sigma1


def findL2(x, V, V0, rho, options):
    # print('finding L2')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    # env = dict(zip(x, np.array([1, 2.31])))
    # print('V0 is %s' % (V0.Evaluate(env)))
    # print('V is %s' % (V.Evaluate(env)))
    # % construct multipliers for Vdot
    L2 = prog.NewFreePolynomial(Variables(x), options.degL2).ToExpression()
    # % construct slack var
    slack = prog.NewContinuousVariables(1, "s")[0]
    prog.AddConstraint(slack >= 0)
    # add normalizing constraint
    prog.AddSosConstraint(-(V - 1) + L2 * (V0 - rho))
    prog.AddSosConstraint(L2)
    prog.AddCost(slack)

    solver = MosekSolver()
    solver.set_stream_logging(False, "")
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    L2 = (result.GetSolution(L2))
    # print(L2.Evaluate(env))
    return L2


def optimizeV(x, f, L1, L2, V0, sigma1, options):
    # print('finding V')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    # env = dict(zip(x, np.array([1, 2.31])))

    #% construct V
    V = prog.NewFreePolynomial(Variables(x), options.degV).ToExpression()

    # V = prog.NewSosPolynomial(Variables(x), options.degV)[0].ToExpression()
    Vdot = V.Jacobian(x) @ f
    # % construct rho
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    # % setup SOS constraints
    prog.AddSosConstraint(-Vdot + L1 * (V - 1) - sigma1 * V / 2)
    prog.AddSosConstraint(-(V - 1) + L2 * (V0 - rho))
    prog.AddSosConstraint(V)
    # % run SeDuMi/MOSEK and check output
    prog.AddCost(-rho)
    solver = MosekSolver()
    solver.set_stream_logging(False, "")
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    V = result.GetSolution(V)
    # print(clean(V))
    rho = result.GetSolution(rho)
    # print('rho is %s' % (rho))
    return V, rho


def levelset_sos(x, V0, f, options):
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    if options.do_balance:
        [T, V, f, _, _] = balance(x, V0, f, None, None)
    else:
        T, V, f = np.eye(options.nX), V0, f
    Vdot = (V.Jacobian(x) @ f)

    H = Jacobian(Vdot.Jacobian(x).T, x)
    env = dict(zip(x, np.zeros(x.shape)))
    H = .5 * np.array([[i.Evaluate(env) for i in j]for j in H])
    print('Hessian of Vdot %s' % (eig(H)[0]))
    assert (np.all(eig(H)[0] <= 0))
    # % construct slack var
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)
    L1 = prog.NewFreePolynomial(Variables(x), options.degL1).ToExpression()

    deg = int(np.floor((options.degL1 + Polynomial(Vdot, x).TotalDegree() -
                        options.degV) / 2))
    prog.AddSosConstraint((x.T@x)**(deg) * (V - rho) + L1 * Vdot)
    # levelsetPoly = Polynomial((x.T@x)**(deg) * (V - rho) + L1 * Vdot, x)
    # prog.AddEqualityConstraintBetweenPolynomials(candidate, levelsetPoly)
    prog.AddCost(-rho)

    solver = MosekSolver()
    solver.set_stream_logging(True, "")
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    L1 = result.GetSolution(L1)
    rho = result.GetSolution(rho)
    # P = result.GetSolution(P)
    # print('cond # of gram is %s' % np.linalg.cond(P))
    print('rho is %s' % rho)
    V = V / rho
    V = (V.Substitute(dict(zip(x, inv(T) @ x))))
    return V


def levelset_w_feature_transformation(system, gram, g, L1):
    # dirty, hard coded everything
    f = system.sym_f
    x = system.sym_x
    V = Polynomial(system.sym_V, x)
    Vdot = Polynomial(system.sym_Vdot, x)
    V = system.sym_V
    Vdot = system.sym_Vdot
    # xxd = Polynomial(system.sym_xxd,x)

    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    # % construct slack var
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    basis = MonomialBasis(x, 8)
    candidate = Polynomial()
    P = prog.NewSymmetricContinuousVariables(45, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    for i in range(45):
        candidate.AddProduct((P[i, i]), basis[i] * basis[i])
        for j in range(i + 1, 45):
            candidate.AddProduct(2 * (P[i, j]), basis[i] * basis[j])

    dim_psi = system.sym_psi.shape[0]
    L1 = prog.NewFreePolynomial(Variables(x), 8).ToExpression()
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    # newbasis = np.array([Polynomial(i,x) for i in system.sym_psi.T@g])
    candidate = Polynomial()
    trans_P = g@P@g.T
    # basis =system.sym_psi
    basis = MonomialBasis(x, 8)

    for i in range(dim_psi):
        candidate.AddProduct((trans_P[i, i]), basis[i] * basis[i])
        for j in range(i + 1, dim_psi):
            candidate.AddProduct(2 * (trans_P[i, j]), basis[i] * basis[j])

    # candidateDecomp = system.sym_psi.T@trans_P@system.sym_psi
    # residual = candidateDecomp-candidate.ToExpression()
    levelsetPoly = Polynomial((x.T@x)**(5) * (V - rho) + L1 * Vdot, x)
    # levelsetPoly = (xxd * (V*Polynomial(rho,x)-1) + L1 *(Vdot))
    prog.AddEqualityConstraintBetweenPolynomials(candidate, levelsetPoly)

    prog.AddCost(-rho)
    solver = MosekSolver()
    solver.set_stream_logging(True, "")
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    # L1 = result.GetSolution(L1)
    rho = result.GetSolution(rho)
    print(rho)
    V = V.ToExpression() * rho
    return V


def balance_quad_form(S, P):
    # copied from the old drake, with only syntax swap
    #  Quadratic Form "Balancing"
    #
    #    T = balqf(S,P)
    #
    #  Input:
    #    S -- n-by-n symmetric positive definite.
    #    P -- n-by-n symmetric, full rank.
    #
    #  Finds a T such that:
    #    T'*S*T = D
    #    T'*P*T = D^(-1)

    # if np.linalg.norm(S - S.T, 1) > 1e-8:
        # raise Error('S must be symmetric')
    # if np.linalg.norm(P - P.T, 1) > 1e-8:
        # raise Error('P must be symmetric')
    # if np.linalg.cond(P) > 1e10:
        # raise Error('P must be full rank')

    # Tests if S positive def. for us.
    V = inv(np.linalg.cholesky(S).T)
    [N, l, U] = np.linalg.svd((V.T.dot(P)).dot(V))
    if N.ravel()[0] < 0:
        N = -N
    T = (V.dot(N)).dot(np.diag(np.power(l, -.25, dtype=float)))
    D = np.diag(np.power(l, -.5, dtype=float))
    return T, D


def balance(x, V, f, S, A):
    if S is None:
        H = Jacobian(V.Jacobian(x).T, x)
        env = dict(zip(x, np.zeros(x.shape)))
        S = .5 * np.array([[i.Evaluate(env) for i in j]for j in H])
    if A is None:
        J = Jacobian(f, x)
        env = dict(zip(x, np.zeros(x.shape)))
        A = np.array([[i.Evaluate(env) for i in j]for j in J])
    [T, D] = balance_quad_form(S, (S@A + A.T@S))
    # print('T is %s' % (T))
    # Sbal = (T.T)@(S)@(T)
    Vbal = V.Substitute(dict(zip(x, T@x)))
    fbal = inv(T)@[i.Substitute(dict(zip(x, T@x))) for i in f]
    return T, Vbal, fbal, S, A


def clean(poly, x, tol=1e-9):
    if isinstance(poly, Expression):
        poly = Polynomial(poly, x)
    return poly.RemoveTermsWithSmallCoefficients(tol).ToExpression()


################
def check_residual(system, gram, rho, L, x_val):
    f = system.sym_f
    x = system.sym_x
    V = system.sym_V
    Vdot = system.sym_Vdot
    l_coeffs = L.T
    L1 = l_coeffs@system.sym_sigma
    candidateDecomp = system.sym_psi.T@gram@system.sym_psi
    levelsetPoly = (system.sym_xxd * (V * rho - 1) + L1 * Vdot)[0]
    env = dict(zip(x, x_val.T))
    ratio = levelsetPoly.Evaluate(env) / candidateDecomp.Evaluate(env)
    print(ratio)
    residual = Polynomial((levelsetPoly - candidateDecomp), x)
    residual_coeffs_mapping = residual.monomial_to_coefficient_map()
    coeffs = list(residual_coeffs_mapping.values())
    for i in coeffs:
        print(i)


def recast_poly_back_to_nonlinear(V, CL_sys):
    """Recast the polynomial Lyapunov candidate back to the original coornidate.
    e.g. originally, we might have recast the non-linearity sin(x) as s, so
    that we have a V=x**2+s**2. In order to visualize, we need the inverse 
    mapping, so that the levelset of V can be properly plotted in the 'x' 
    cooridnate

    Args:
        x (Inderterminates):
        V (Inderterminates): V(x)
        CL_sys(ClosedLoopSystem): encodes the mapping
    """
    env = CL_sys.inverse_recast_map()
    nonPolyV = V.Substitute(env)
    return nonPolyV


def IQC_tanh(x, y):
    y_cross = 0.642614
    x_off = .12
    return np.hstack((y ** 2 - 1,
                      (y - ((x) + x_off)) * (y + y_cross),
                      (y - ((x) - x_off)) * (y - y_cross),
                      (y - x) * y))


# def solve_LP_for_V(phi, dphidx, f, num_samples=None):
#     if num_samples is None:
#         num_samples = phi.shape[0]
#     monomial_dim = phi.shape[-1]
#     prog = MathematicalProgram()
#     P = prog.NewSymmetricContinuousVariables(monomial_dim, "P")
#     allv = 0
#     allvdot = 0
#     for i in range(num_samples):
#         this_v = phi[i, :].T@P@phi[i, :]
#         prog.AddConstraint(this_v >= 0)
#         this_vdot = phi[i, :].T@P@dphidx[i, :, :]@f[i, :]
#         prog.AddConstraint(this_vdot <= 0)
#         allv = allv + this_v
#         allvdot = allvdot + this_vdot
#     prog.AddConstraint(phi[1, :].T@P@phi[1, :] == 1)
#     # prog.AddCost(allvdot-allv)
#     # prog.AddCost(allvdot)
#     prog.AddCost(0)
#     solver = MosekSolver()
#     solver.set_stream_logging(False, "")
#     result = solver.Solve(prog, None, None)
#     print(result.get_solution_result())
#     assert result.is_success()
#     P = result.GetSolution(P)
#     print('eig of orignal A  %s' % (eig(P)[0]))
#     # print('eig of orignal SA+A\'S  %s' % (eig(A0.T@S0 + S0@A0)[0]))
#     return P


# def levelsetLP(system, gram):
#     f = system.sym_f
#     x = system.sym_x
#     V = system.sym_V
#     Vdot = system.sym_Vdot
#     prog = MathematicalProgram()

#     # % construct slack var
#     rho = prog.NewContinuousVariables(1, "r")[0]
#     prog.AddConstraint(rho >= 0)
#     scaling = prog.NewContinuousVariables(gram.shape[0], "s")
#     for i in scaling:
#         prog.AddConstraint(i >= 0)
#     slack = prog.NewContinuousVariables(1, "l")[0]

#     l_coeffs = prog.NewContinuousVariables(system.sym_sigma.shape[0], "L")
#     L1 = l_coeffs@system.sym_sigma
# candidateDecomp =
# Polynomial(system.sym_psi.T@np.diag(scaling)@gram@system.sym_psi, x)

#     levelsetPoly = Polynomial(
#         system.sym_xxd * (V - rho) + L1 * Vdot + slack, x)
#     prog.AddEqualityConstraintBetweenPolynomials(candidateDecomp, levelsetPoly)
#     prog.AddCost(-rho)
#     solver = MosekSolver()
#     solver.set_stream_logging(True, "")
#     result = solver.Solve(prog, None, None)
#     print(result.get_solution_result())
#     # print('w/ solver %s' % (result.get_solver_id().name()))
#     assert result.is_success()
#     L1 = result.GetSolution(L1)
#     rho = result.GetSolution(rho)
#     print(rho)
#     V = V / rho
#     V = (V.Substitute(dict(zip(x, inv(T) @ x))))
#     return V
