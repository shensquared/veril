import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plotFunnel(x, V):
  xlim = np.load('/Users/shenshen/SafeDNN/data/VanDerPol_limitCycle.npy')
  fig, ax = plt.subplots()
  bdry = ax.plot(xlim[0, :], xlim[1, :], label='ROA boundary')
  plt.scatter(samples[:, 0], samples[:,1])
  plt.show()
  # coords = obj.getFrame.getCoordinateNames()
  # xlabel(coords{options.plotdims(1)})
  # ylabel(coords{options.plotdims(2)})


# def getLevelSet(x, f, options):
#   '''
# # return points on the (first) level set f(x)==1 surrounding x0
# #
# # @param x a simple msspoly defining the coordinates
# # @param f an msspoly of the scalar function you want to plot
# # @option x0 the fixed point around which the function will be plotted.  @
# # default 0
# # @option num_samples number of samples to produce @default 100
# # @option tol
# # @option plotdims restrict level set points to a subset of the dimensions.
# '''
#   # if (nargin<3) options=struct(); end
#   # if (~isfield(options,'tol')) options.tol = 2e-3; end # default tolerance of fplot
#   # if ~isfield(options,'num_samples') options.num_samples = 100; end
#   # if ~isfield(options,'x0') options.x0 = zeros(size(x)); end

#   # if isfield(options,'plotdims')
#   #   no_plot_dims=1:length(options.x0);  no_plot_dims(options.plotdims)=[];
#   #   if ~isempty(no_plot_dims)
#   #     f = subs(f,x(no_plot_dims),options.x0(no_plot_dims));
#   #     x = x(options.plotdims);
#   #     options.x0 = options.x0(options.plotdims);
#   #   end
#   # end

#   if (deg(f, x) <= 2):
#     # # interrogate the quadratic level-set
#     # # note: don't need (or use) x0 in here

#     # # f = x.T@A@x + b.T@x + c
#     # # dfdx = x'(A+A') + b'
#     # # H = .5*(A+A')   # aka, the symmetric part of A
#     # #   dfdx = 0 => xmin = -(A+A')\b = -.5 H\b
#     H = 0.5 * Jacobian(f.Jacobian(x).T, x)
#     # if ~isPositiveDefinite(H), error('quadratic form is not positive
#     # definite')
#     xmin = -0.5 * (H/(subs(diff(f, x), x, 0 * x)'))
#     fmin = doubleSafe(subs(f, x, xmin));
#     if (fmin > 1)
#       error('minima is >1');

#     n = length(x);
#     K = options.num_samples;
#     if (n == 2)  # produce them in a nice order, suitable for plotting
#       th = linspace(0, 2 * pi, K);
#       X = [sin(th); cos(th)];
#     else
#       X = randn(n, K);
#       X = X. / repmat(sqrt(sum(X. ^ 2, 1)), n, 1);
#     end

#     # f(x) = fmin + (x-xmin)'*H*(x-xmin)
#     #   => 1 = fmin + (y-xmin)'*H*(y-xmin)
#     y = repmat(xmin, 1, K) + (H / (1 - fmin)) ^ (-1 / 2) * X;
#   else  # do the more general thing

#     if (length(x) ~=2) error('not supported yet'); end

#     # assume star convexity (about x0).
#     if (double(subs(f, x, options.x0)) > 1)
#       error('x0 is not in the one sub level-set of f');
#     end

#     f = subss(f, x, x + options.x0);  # move to origin

#     y = repmat(options.x0, 1, options.num_samples) + getRadii(linspace(-pi, pi, options.num_samples))';
#     # #  r = msspoly('r',1);
#     # #   [theta,r]=fplot(@getRadius,[0 pi],options.tol);
#     # #   y=repmat(x0,1,2*size(theta,1))+[repmat(r(:,1),1,2).*[cos(theta),sin(theta)]; repmat(r(:,2),1,2).*[cos(theta),sin(theta)]]';

#   if (any(imag(y(:))))
#     error('something is wrong.  i got imaginary outputs');

#   # # Assumes that the function is radially monotonic.  This could
#   # # break things later.


# def getRadii(thetas):  # needs to be vectorized
#       rU = ones(size(thetas));
#       rL = zeros(size(thetas));
#       CS = [cos(thetas); sin(thetas)];
#       evaluate = @(r) double(msubs(f, x, repmat(r, 2, 1).*CS));
#       msk = evaluate(rU) < 1;
#       while any(msk)
#           rU(msk) = 2 * rU(msk);
#           msk = evaluate(rU) < 1;
#       end

#       while (rU - rL) > 0.0001 * (rU + rL)
#           r = (rU + rL) / 2;
#           msk = evaluate(r) < 1;
#           rL(msk) = r(msk);
#           rU(~msk) = r(~msk);
#       end

#       y = (repmat(r, 2, 1).*CS)'
