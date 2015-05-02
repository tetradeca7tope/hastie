function [K] = PolyKernel(degree, bias, X, Y)
% X (n, d)
% Y if exists (ny, d)
% K (n,n) or (n,ny)
  if ~exist('Y', 'var') | isempty(Y)
    Y = X;
  end

   [n, d] = size(X);
   K = (X * Y' + bias) .^ degree;
end
