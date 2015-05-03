function [kernelFunc, decomposition, bandwidths, scales] = ...
  kernelSetup(X, Y, decomposition)
% This is a wrapper for combinedKernel. It returns a function handle which can
% be used to compute the Kernels directly from the data. This function takes
% care of the bandwidths etc.
% X, Y: covariates and labels. As is, the Y's aren't really used but passing
%       them here in case we need to design kernels (later on) using Y.
% decomposition: A struct which contains info on how to construct the
%   decomposition. Read obtainDecomposition.

  % Prelims
  n = size(X, 1);
  numDims = size(X, 2);

  if strcmp(decomposition.setting, 'espKernel')
    % In this case we use the newton-girard trick and elementary symmetric
    % polynomials to compute all k^th order interactiosn for k = 1:D
    bws = 20 * std(X) * n^(-1/5);
%     bws = 1.5 * std(X) * n^(-1/5);
%     bws = 1.5 * norm(std(X)) * n^(-1/(4+numDims)) * ones(numDims, 1);
    kernelFunc = @(X1, X2) espKernels(X1, X2, bws);
    decomposition.M = numDims;

  else 
    % Obtain the decomposition
    decomposition = obtainDecomposition(numDims, decomposition);
    groups = decomposition.groups;
    M = numel(groups);
    decomposition.M = M;

    % Kernel Scales and bandwidths
    scales = ones(M, 1); % Just use all 1s
    dimStds = std(X);
    for j = 1:M
      coords = groups{j};
      numGroupDims = numel(coords);
      bandwidths(j) = 1.5 * norm(dimStds(coords)) * n^(-1/(4 + numGroupDims));
    end

    kernelFunc = @(X1, X2) combinedKernel(X1, X2, groups, bandwidths, scales);
  end

end

