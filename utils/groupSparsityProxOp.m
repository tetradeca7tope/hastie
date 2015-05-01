function Z = groupSparsityProxOp(X, tVals)
% Obtains the Prox Operator for Prox Operator on the sum of columnwise l2-norm
% penalties of X. I.e h(X) = \sum_j tVals(j)*||X(:,j)||_2 where X(:,j) \in R^n.
% X is an n x numGroups matrix. 
% tVals is a vector of t-values for each group. If tVals is a scalar we will use
% this value for all groups.

  n = size(X, 1);
  numGroups = size(X, 2);
%   Z = sparse(n, numGroups);
  Z = zeros(n, numGroups);
  if isscalar(tVals), tVals = tVals * ones(numGroups, 1);
  end

  for j = 1:numGroups
    Z(:, j) = l2NormProxOp(X(:,j), tVals(j));
  end

end

