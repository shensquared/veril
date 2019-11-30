import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (Polynomial, Variable, Variables, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)


def sample_on_variety(variety, num_attempts):
    """returns the pure 'x' samples on one given variety. i.e. return the roots
    to the given multi-variate polynomial. Note that, due to the numerical
    accuracy of numpy's polyroots, only returning all the real roots.
    Args:
        variety (Pydrake expression or polynomial): the variety-defining
        polynomial
        num_attempts (int): number of 'attempts' of turning the multi-variate
        into univarite, i.e., the final num_roots != num_attempts. (since
        there're d complex roots for a univariate polynomial of degree d, and
        thus more than 1 real root)

    Returns:
        n_roots-by-num_states(num_x in the variety): all real roots for the
        univarite polynomial coming from the num_attempts attemps of
        transformation
    """
    x = np.array(list(variety.GetVariables()))
    nx = x.shape[0]
    t = Variable('t')
    samples = np.zeros((2,))
    for i in range(num_attempts):
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


def sample_to_monomials(system, variety, num_samples, do_transform=False):
    enough_samples = False
    sufficient_samples = 0
    while not enough_samples:
        samples = sample_on_variety(variety, num_samples + sufficient_samples)
        [V, Vdot, xxd, psi, sigma] = system.get_levelset_features(samples)
        if do_transform:
            psi, T, n = coordinate_ring_transform(psi)
        enough_samples, sufficient_num_samples = prune_sample(psi)
        print('sample rank is %s' % sufficient_num_samples)

    return [V[:sufficient_num_samples], Vdot[:sufficient_num_samples], xxd
            [:sufficient_num_samples], psi[:sufficient_num_samples, :], sigma
            [:sufficient_num_samples, :]]


def coordinate_ring_transform(sampled_monomials):
    """reduce the dimensionality of the sampled-monimials by taking advantage
    of the coordiate ring structure (similar to Gaussian elimination used in
    Grobner basis)

    Args:
        sampled_monomials: (num_samples, monomial_dim)
        U =sampled_monomials.T (monomial_dim, num_samples)
        [u,s,v] = svd(U)
        n = # of non-zero values in s
        U= T@U_transformed, where
        T:(monomial_dim, n),
        U_transformed:(n,num_samples)

    Returns:
        U_transformed.T (num_samples, reduced_monomials)
    """
    U = sampled_monomials.T
    [u, diag_s, v] = np.linalg.svd(U)
    tol = max(U.shape) * diag_s[0] * 1e-16
    n = sum(diag_s > tol)
    s = np.zeros(U.shape)
    np.fill_diagonal(s, diag_s)
    U_transformed = v[:n, :]
    T = u@s[:, :n]
    return U_transformed.T, T, n


def prune_sample(all_samples):
    """returns the number of samples necessary

    Args:
        all_samples (ndarry): (m,n), current samples,
        m (int): current number of samples
        n (int): monomial_dim (or reduced_monomial if do_transformation)
    """
    enough_samples = True
    m, n = all_samples.shape
    n2 = n * (n + 1) / 2
    m0 = min(m, n2)
    sub_samples = all_samples[:m0, :]

    c = np.power(sub_samples@sub_samples.T, 2)  # c = q'*q
    s = abs(np.linalg.eig(c)[0])
    tol = max(c.shape) * np.spacing(max(s))
    sample_rank = sum(s > tol)
    if sample_rank == m0 and sample_rank < n2:
        enough_samples = False
        print('Insufficient samples!!')
    return enough_samples, m0


def solve_SDP_on_samples(system, sampled_quantities):
    prog = MathematicalProgram()
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    [V, Vdot, xxd, psi, sigma] = sampled_quantities
    dim_psi = psi.shape[1]
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    for i in range(psi.shape[0]):
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
    idx = []
    isVanishing = True
    for i in range(test_samples.shape[0]):
        levelset = (xxd[i] * (V[i] - rho))[0]
        candidate = psi[i].T@P@psi[i]
        ratio = levelset / candidate
        print('two polynomials evals ratio %s' % ratio)
        if abs(ratio-1) > 1e-2:
            isVanishing = False
            idx += [i]
    return isVanishing, [V[idx], Vdot[idx], xxd[idx], psi[idx], sigma[idx]]
