import numpy as np


def linspace_data(d=10, num_grid=100):
    # TODO: hard-coded 2-dimensional, need to be general
    x1 = np.linspace(-d, d, num_grid)
    x2 = np.linspace(-d, d, num_grid)
    x1 = x1[np.nonzero(x1)]
    x2 = x2[np.nonzero(x2)]
    x1, x2 = np.meshgrid(x1, x2)
    x1, x2 = x1.ravel(), x2.ravel()
    return [np.array([x1, x2]).T, np.zeros(x1.shape)]


def withinLevelSet(V):
    x = list(V.GetVariables())
    y, max_r, min_r = levelsetData(V)
    samples = linspace_data(max_r, num_grid=200)[0]
    in_points = np.zeros((1, 2))
    # out_points = np.zeros((1, 2))
    for s in samples:
        env = dict(zip(x, s.T))
        if V.Evaluate(env) <= 1:
            in_points = np.vstack((in_points, s))
        # else:
            # out_points = np.vstack((out_points, s))
    in_points = in_points[1:, :]
    # out_points = out_points[1:, :]
    return in_points, np.zeros(in_points.shape[0])


def levelsetData(V, num_grid=100):
    # TODO  % move to origin
    # f=subss(f,x,x+options.x0);
    # TODO if degree 2, quadratic, use simpler stuff
    # if (deg(f,x)<=2)  % interrogate the quadratic level-set
    #   % note: don't need (or use) x0 in here

    #   % f = x'Ax + b'x + c
    #   % dfdx = x'(A+A') + b'
    #   % H = .5*(A+A')   % aka, the symmetric part of A
    #   %   dfdx = 0 => xmin = -(A+A')\b = -.5 H\b
    #   H = doubleSafe(0.5*diff(diff(f,x)',x));
    #   if ~isPositiveDefinite(H), error('quadratic form is not positive definite'); end
    #   xmin = -0.5*(H\doubleSafe(subs(diff(f,x),x,0*x)'));
    #   fmin = doubleSafe(subs(f,x,xmin));
    #   if (fmin>1)
    #     error('minima is >1');
    #   end

    #   n=length(x);
    #   K=options.num_samples;
    #   if (n==2)  % produce them in a nice order, suitable for plotting
    #     th=linspace(0,2*pi,K);
    #     X = [sin(th);cos(th)];
    #   else
    #     X = randn(n,K);
    #     X = X./repmat(sqrt(sum(X.^2,1)),n,1);
    #   end

    #   % f(x) = fmin + (x-xmin)'*H*(x-xmin)
    #   %   => 1 = fmin + (y-xmin)'*H*(y-xmin)
    #   y = repmat(xmin,1,K) + (H/(1-fmin))^(-1/2)*X;
    y, max_r, min_r = getRadii(np.linspace(-np.pi, np.pi, num_grid), V)
    y = np.tile(np.zeros((2, 1)), (1, num_grid)) + y.T
    return y.T, max_r, min_r

# Assumes that the function is radially monotonic.  This could
# break things later.


def getRadii(thetas, V):  # needs to be vectorized
    x = list(V.GetVariables())
    n = thetas.shape[0]
    rU = np.ones(thetas.shape)
    rL = np.zeros(thetas.shape)
    CS = np.vstack((np.cos(thetas), np.sin(thetas)))

    evaluate = lambda r: np.array([V.Evaluate(dict(zip(x, (np.tile(r, (2, 1)) *
                                                           CS)[:, i]))) for i in range(n)])

    msk = evaluate(rU) < 1
    while any(msk):
        rU[msk] = 2 * rU[msk]
        msk = evaluate(rU) < 1

    while all((rU - rL) > 0.0001 * (rU + rL)):
        r = (rU + rL) / 2
        msk = evaluate(r) < 1
        rL[msk] = r[msk]
        rU[~msk] = r[~msk]

    y = (np.tile(r, (2, 1)) * CS).T
    return y, max(r), min(r)
