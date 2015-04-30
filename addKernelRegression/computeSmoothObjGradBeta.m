function [obj, G] = computeSmoothObjGradBeta(Beta, Ls, Y)
% Computes the value of the smooth part of the objective and its gradient if
% requested.
% Beta is an nxM matrix where each column is alpha_j
% Ls is an nxnxM tensor with Ks(:,:,j) corresponding to the Lower triangular
% cholesky decomposition of the kernel matrix of the j^th kernel.
% Y are the labels.

  % Prelims
  [n, M] = size(Beta);

  % Compute the following
  LBeta = zeros(n,M);
  for j = 1:M
    LBeta(:,j) = Ls(:,:,j) * Beta(:,j);
  end
  diff = sum(LBeta, 2) - Y; % The difference between the predictions and labels.

  % The objective
  obj = 1/(2*n) * norm(diff)^2;

  % Compute gradient if requested
  if nargout == 2
    G = zeros(n, M);
    for j = 1:M
      G(:,j) = (1/n) * Ls(:,:,j)' * diff;
    end

  elseif nargout > 2
    error('Too many output arguments.');
  end

end

