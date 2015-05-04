function [kernelFunc, decomposition] = ...
  kernelSetupPoly(X, Y, decomposition, params)
% This is a wrapper for combinedKernelPoly. It returns a function handle which can
% be used to compute the Kernels directly from the data. This function takes
% care of the bandwidths etc.
% X, Y: covariates and labels. As is, the Y's aren't really used but passing
%       them here in case we need to design kernels (later on) using Y.
% decomposition: A struct which contains info on how to construct the
%   decomposition. Read obtainDecomposition.

  % Prelims
  n = size(X, 1);
  numDims = size(X, 2);

  % Obtain the decomposition
  decomposition = obtainDecomposition(numDims, decomposition);
  groups = decomposition.groups;
  M = numel(groups);

  kernelFunc = @(X1, X2) combinedKernelPoly(X1, X2, groups, params);

end

function [K, allKs] = combinedKernelPoly(X1, X2, groups, params)
% This computes all sub kernels and the sum Kernel K.
% groups is a numGroups size 

  numGroups = numel(groups);
  n1 = size(X1, 1);
  n2 = size(X2, 1);
  K = zeros(n1, n2);
  allKs = zeros(n1, n2, numGroups);

  for k = 1:numGroups
    coords = groups{k};
    allKs(:,:,k) = subKernelPoly(X1, X2, coords, params);
  end

  % Now sum all the Kernels.
  K = sum(allKs, 3);

end

function K = subKernelPoly(X1, X2, coords, params)
% Each of the small kernels is only affected by a subset of the coordinates.
% So we need to make sure that the output of the kernel only depends on these
% quantities. This is what this function is doing.

  % Sometimes we get the entire X (with all coordinates) sometimes we only get
  % the relevant coordinates. Check for this condition.
  if size(X1, 2) == numel(coords), X1sub = X1;
  else, X1sub = X1(:, coords);
  end
  if size(X2, 2) == numel(coords), X2sub = X2;
  else, X2sub = X2(:, coords);
  end

  K = (X1sub * X2sub' + params.bias) .^ params.degree;
end

