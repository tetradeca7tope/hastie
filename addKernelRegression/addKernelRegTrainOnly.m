function [predFunc, optAlpha, optBeta, optStats, decomposition] = ...
  addKernelRegTrainOnly(X, Y, decomposition, lambda, params)
% This is a wrapper for addKernelRegOpt - it constructs kernels and obtains
% their cholesky decomposition before calling addKernelRegOpt. It does not do
% any cross validation. I am writing this function because it is useful to
% debug/ test the optimisation procedures and also perform comparisons in
% between them.

% X, Y: Covariates and Labels
% decomposition is a structure containing the groups.
% lambda: penalty coefficient for regularization.

  % prelims
  n = size(X, 1);

  % Obtain the Kernels
  [kernelFunc, decomposition] = kernelSetup(X, Y, decomposition);
  groups = decomposition.groups;
  M = numel(groups);
  [~, allKs] = kernelFunc(X, X); % Compute the kernel
  % Now obtain the cholesky Decompositions
  allLs = zeros(size(allKs));
  for j = 1:M
    allLs(:,:,j) = stableCholesky(allKs(:,:,j));
  end

  % Set initialisation
  initBeta = zeros(n, M);

  % Now call addKernelRegOpt
  [optBeta, optStats] = addKernelRegOpt(allLs, Y, decomposition, ...
    lambda, initBeta, params);

  % Now return the predictor function
  optAlpha = zeros(n, M);
  for j = 1:M
    optAlpha(:,j) = (allLs(:,:,j)') \ optBeta(:,j);
  end
  predFunc = @(Xte) getPrediction(Xte, X, optAlpha, kernelFunc);

end

