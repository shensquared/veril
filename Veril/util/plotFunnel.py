def plotFunnel(obj,options)
'''
% Plots the one-level set of V
%
% @param options options structure
%
% @option plotdims coordinates along which to plot given as a 1x2 array
%   @default [1, 2]
% @option x0 default coordinates (e.g. to be used in the slice)
% @option inclusion = { 'slice' | 'projection' }
%    'slice' -- include the points in the plane of the given
%        dimensions for which V(x) < 1.
%    'projection' -- plot the projection of {x | V(x) < 1} into the
%        given plane.
% @option color matlab color for the funnel @default [.7 .7 .7]
% @option tol tolerance for computing the level set
%
% @retval h column vector of handles for any graphics objects created

% todo: support wrapping coordinates

if (nargin<2) options=struct(); end
if ~isfield(options,'plotdims') options.plotdims = [1;2]; end
if ~isfield(options,'x0') options.x0 = zeros(obj.getFrame.dim,1); end  
if (~isfield(options,'inclusion')) options.inclusion = 'slice'; end
if (~isfield(options,'color')) options.color=.7*[1 1 1]; end
if (~isfield(options,'tol')) options.tol = .01; end
'''
x=getLevelSet(obj,0,options);
h=plt.fill(x(1,:),x(2,:),repmat(0,1,size
  (x,2)),options.color,'LineStyle','-','LineWidth',2);
coords = obj.getFrame.getCoordinateNames();
xlabel(coords{options.plotdims(1)})
ylabel(coords{options.plotdims(2)})



