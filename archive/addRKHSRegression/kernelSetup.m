function [kernelFunc, decomposition, bandwidths, scales] = ...
  kernelSetup(X, Y, decomposition)
% This is a wrapper for combinedKernel. It returns a function handle which can
% be used to compute the Kernels directly from the data. This function takes
% care of the bandwidths etc.

  % Prelims
  n = size(X, 1);
  numDims = size(X, 2);

  % Obtain the decomposition
  decomposition = obtainDecomposition(numDims, decomposition);
  groups = decomposition.groups;
  M = numel(groups);

  % Kernel Scales and bandwidths
  scales = 2 * std(Y)/sqrt(M) * ones(M, 1);
  dimStds = std(X);
  for j = 1:M
    coords = groups{j};
    numGroupDims = numel(coords);
    bandwidths(j) = 1.5 * norm(dimStds(coords)) * n^(-1/(4 + numGroupDims));
  end

  kernelFunc = @(X1, X2) combinedKernel(X1, X2, groups, bandwidths, scales);

end

