import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (Polynomial, Variable, Variables, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)


def sample_on_variety(variety, root_threads):
    """returns the pure 'x' samples on one given variety. i.e. return the roots
    to the given multi-variate polynomial. Note that, due to the numerical
    accuracy of numpy's polyroots, only returning all the real roots.
    Args:
        variety (Pydrake expression or polynomial): the variety-defining
        polynomial
        root_threads (int): number of 'attempts' of turning the multi-variate
        into univarite, i.e., the final num_roots != root_threads. (since
        there're d complex roots for a univariate polynomial of degree d, and
        thus more than 1 real root)

    Returns:
        n_roots-by-num_states(num_x in the variety): all real roots for the
        univarite polynomial coming from the root_threads attemps of
        transformation
    """
    x = np.array(list(variety.GetVariables()))
    nx = x.shape[0]
    t = Variable('t')
    samples = np.zeros((nx,))
    num_roots = 0
    while num_roots < root_threads:
        # alphas = np.random.uniform(-2, 2, nx)
        # betas = np.random.uniform(-2, 2, nx)
        alphas = np.random.randn(nx)
        betas = np.random.randn(nx)

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
        n_root_x = root_x.shape[0]
        root_x_norm = np.array([np.linalg.norm(i) for i in root_x])
        root_x = np.array([root_x[i] for i in range(n_root_x) if root_x_norm
        [i]>1e-4])

        varitey_x = np.array([variety.Evaluate(dict(zip(x, i))) for i in
            root_x])
        V_x = np.array([variety.Evaluate(dict(zip(x, i)))
                              for i in root_x])
        close = np.isclose(varitey_x, 0, atol=4e-12)

        if np.any(close):
            root_x = root_x[close]
            samples = np.vstack([samples, root_x])
            num_roots = samples.shape[0] - 1
    # print(samples)
    return samples[1:, :]


def sample_monomials(system, samples, variety,  do_transform=True):
    enough_samples = False
    while not enough_samples:
        new_samples = sample_on_variety(
            variety, int(np.ceil(samples.shape[0] * .01)))
        samples = np.vstack([samples, new_samples])
        [V, xxd, psi] = system.get_sample_variety_features(samples)
        balanced = max(V)[0]/min(V)[0]<1e3
        while not balanced:
            print('not balanced')
            print(V)
            idx = [np.argmax(V), np.argmin(V)]
            samples=np.delete(samples,idx,axis=0)
            V=np.delete(V,idx,axis=0)
            xxd=np.delete(xxd,idx,axis=0)
            psi=np.delete(psi,idx,axis=0)
            balanced = max(V)[0]/min(V)[0]<1e3
        if do_transform:
            trans_psi, Tinv = coordinate_ring_transform(psi)
        else:
            Tinv = np.eye(psi.shape[1])
        enough_samples = check_genericity(trans_psi)
    return [V, xxd, trans_psi], Tinv


def coordinate_ring_transform(monomial_samples):
    """reduce the dimensionality of the sampled-monimials by taking advantage
    of the coordiate ring structure (similar to Gaussian elimination used in
    Grobner basis)

    Args:
        monomial_samples: (num_samples, monomial_dim)
        U =monomial_samples.T (monomial_dim, num_samples)
        [u,s,v] = svd(U)
        n = # of non-zero values in s
        U= T@U_transformed, where
        T:(monomial_dim, n),
        U_transformed:(n,num_samples)
        for testing, standard_monomial = T * reduced_basis, or
        pinv(T)@standard_monomial = reduced_basis
    Returns:
        U_transformed.T (num_samples, reduced_monomials)
    """
    U = monomial_samples.T
    [u, diag_s, v] = np.linalg.svd(U)
    tol = max(U.shape) * diag_s[0] * 1e-16
    original_monomial_dim = U.shape[0]
    n = sum(diag_s > tol)
    if n/original_monomial_dim >= .95:
        print('no transforming')
        return U.T, np.eye(original_monomial_dim)
    else:
        s = np.zeros(U.shape)
        np.fill_diagonal(s, diag_s)
        U_transformed = v[:n, :]
        T = u@s[:, :n]
        Tinv = np.linalg.pinv(T)
        return U_transformed.T, Tinv

# U=np.array([[1,1,1,0,0,0],[-0.6, -1.2, -0.75, 0.8,0.4,0.25],[-1.2,-.6,-.75,-.4,-.8,-.25],[1.2,0.6,.75,.4,.8,.25],[-.6,-1.2,-.75,.8,.4,.25]])
# coordinate_ring_transform(U.T)


def check_genericity(all_samples):
    """check the rank condition of the samples to make sure the genericity is
    satisfied

    Args:
        all_samples (ndarry): (m,n), current samples,
        m (int): current number of samples
        n (int): monomial_dim (or reduced_monomial if do_transformation)
    Returns:
        enough_samples (Bool): if the current sample set is generic enough
        m0 (int): current samples rank, gives good indicator of whether to
        augment or truncate the current sample set
        diff_m (int): m-m0
    """
    enough_samples = True
    m, n = all_samples.shape
    n2 = n * (n + 1) / 2
    m0 = min(m, n2)
    sub_samples = all_samples[:m0, :]

    c = np.power(sub_samples@sub_samples.T, 2)  # c = q'*q
    s = abs(np.linalg.eig(c)[0])
    tol = max(c.shape) * np.spacing(max(s)) * 1e3
    sample_rank = sum(s > tol)
    if sample_rank == m0 and sample_rank < n2:
        # meaning m<n2 and sample full rank, use huristic and return an
        # increase of 30%
        # print('Insufficient samples!!')
        enough_samples = False
        # diff_m >0, increase subsequent sampling
        # diff_m = np.ceil((n2 - m) / 2)
    # sample_rank by construction less than m0, so unless the previous if is
    # true, diff_m < 0, can do subsequent down-sample
    return enough_samples


def solve_SDP_on_samples(system, sampled_quantities):
    prog = MathematicalProgram()
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    [V, xxd, psi] = sampled_quantities
    print('SDP V %s' %V)
    dim_psi = psi.shape[1]
    print('SDP size is %s' % dim_psi)
    # print('num SDP is %s' % psi.shape[0])
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    for i in range(psi.shape[0]):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        prog.AddConstraint(residual[0] == 0)

    prog.AddCost(-rho)
    solver = MosekSolver()
    solver.set_stream_logging(False, "")
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    P = result.GetSolution(P)
    rho = result.GetSolution(rho)
    # print(rho)
    V = system.sym_V / rho
    return V, rho, P


def check_vanishing(system, rho, P, Tinv):
    test_samples = sample_on_variety(system.sym_Vdot, 2)
    [V, xxd, psi] = system.get_sample_variety_features(test_samples)
    idx = []
    isVanishing = True
    for i in range(test_samples.shape[0]):
        print('vanishing V %s' %V)
        levelset = (xxd[i] * (V[i] - rho))[0]
        this_psi = Tinv@psi[i]
        candidate = this_psi.T@P@this_psi
        # order = np.ceil(np.log(candidate))
        # print(order)
        ratio = levelset / candidate
        print('two polynomials evals ratio %s' % ratio)
        if ratio <.97 or ratio > 1.1:
            isVanishing = False
            idx += [i]
    return isVanishing, test_samples[idx]
