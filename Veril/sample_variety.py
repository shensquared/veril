import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (Polynomial, Variable, Variables, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)


def sample_on_variety(variety, num_samples):
    x = np.array(list(variety.GetVariables()))
    nx = x.shape[0]
    t = Variable('t')
    samples = np.zeros((2,))
    for i in range(num_samples):
        alphas = np.random.uniform(-1, 1, nx)
        betas = np.random.uniform(-1, 1, nx)
        parameterization = alphas * t + betas
        env = dict(zip(x, parameterization))
        Vt = variety.Substitute(env)
        Vt = Polynomial(Vt, [t])
        coeffs = [i.Evaluate({t: 0}) for i in list(
            Vt.monomial_to_coefficient_map().values())]
        root_t = np.polynomial.polynomial.polyroots(coeffs)
        # need only the real roots
        root_t = root_t[~np.iscomplex(root_t)].real
        root_x = np.array([alphas * i + betas for i in root_t])
        varitey_x = np.array([variety.Evaluate(dict(zip(x, i)))
                              for i in root_x])
        root_x = root_x[np.isclose(varitey_x, 0, atol=4e-12), :]
        samples = np.vstack([samples, root_x])
    return samples[1:, :]


def solve_SDP_on_samples(system, samples):
    prog = MathematicalProgram()

    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    dim_psi = system.sym_psi.shape[0]
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    [V, Vdot, xxd, psi, sigma] = system.get_levelset_features(samples)

    for i in range(samples.shape[0]):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        prog.AddConstraint(residual[0] == 0)

    prog.AddCost(-rho)
    solver = MosekSolver()
    solver.set_stream_logging(True, "")
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    assert result.is_success()
    P = result.GetSolution(P)
    rho = result.GetSolution(rho)
    print(rho)
    V = system.sym_V / rho
    return V, rho, P


def check_vanishing(system, rho, P):
    test_samples = sample_on_variety(system.sym_Vdot, 1)
    [V, Vdot, xxd, psi, sigma] = system.get_levelset_features(test_samples)
    for i in range(test_samples.shape[0]):
        levelset = (xxd[i] * (V[i] - rho))[0]
        candidate = psi[i].T@P@psi[i]
        assert(np.isclose(levelset, candidate, rtol=1e-04))
