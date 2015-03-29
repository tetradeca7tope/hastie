function [obj, G] = computeSmoothObjGrad(Alpha, Ks, Y, lambda1)
% Coomputes the value of the smooth part of the objective and its gradient if
% requested.
% Alpha is an nxM matrix containing the alpha values
% Ks is an nxnxM tensor with Ks(:,:,j) corresponding to the Kernel matrix of the
% j^th kernel.
% Y are the labels
% lambda1 is the regularization parameter for the RKHS penalty.
 
  % Prelims
  n = size(Alpha, 1);
  M = size(Alpha, 2);

  % Compute the following
  KAlpha = zeros(n, M);
  for j = 1:M
    KAlpha(:,j) = Ks(:,:,j) * Alpha(:, j);
  end
  diff = sum(KAlpha, 2) - Y; % The difference between predictions and labels.

  % obtain the objective
  obj = 0.5*norm(diff)^2 + 0.5*lambda1 * sum(sum(Alpha .* KAlpha));
  
  if nargout == 2
    G = zeros(n, M); % This won't be sparse.
    for j = 1:M
      G(:,j) = Ks(:,:,j) * diff + lambda1 * KAlpha(:,j);
    end
  elseif nargout > 2
    error('Too many output arguments.');
  end

end

