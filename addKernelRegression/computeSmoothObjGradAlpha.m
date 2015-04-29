function [obj, KAlpha, G] = computeSmoothObjGradAlpha(Alpha, Ks, Y)
% Computes the value of the smooth part of the objective and its gradient if
% requested.
% Alpha is an nxM matrix where each column is alpha_j
% Ks is an nxnxM tensor with Ks(:,:,j) corresponding to the kernel matrix of the
% j^th kernel.
% Y are the labels.

  % Prelims
  n = size(Alpha, 1);
  M = size(Alpha, 2);

  % Compute the following
  KAlpha = zeros(n, M);
  for j = 1:M
    KAlpha(:,j) = Ks(:,:,j) * Alpha(:,j);
  end
  diff = sum(KAlpha,2) - Y;

  % The objective
  obj = 1/(2*n) * norm(diff)^2;

  % The gradient if requested
  if nargout == 3
    G = zeros(n, M);
    for j = 1:M
      G(:,j) = (1/n) * Ks(:,:,j) * diff;
    end
  elseif nargout > 3
    error('Too many output arguments.');
  end

end

