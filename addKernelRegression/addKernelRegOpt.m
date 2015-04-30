function [optBeta, optStats, decomposition] = ...
  addKernelRegOpt(Ls, Y, decomposition, lambda, initPt, params)
% Performs Additive Kernel Regression.
% Ks, Y: All Kernels Covariates and Labels
% decomposition is a struct containing the groups.
% lambda: penalty coefficient.
% params: contains ancillary params

% N.B: This function trains using all of X, Y for the given value of lambda. 
% Ideally we should be using CV etc for which we will write a wrapper.

  % prelims
  n = size(Y, 1);
  groups = decomposition.groups;
  M = numel(groups);

  if ~isfield(params, 'optVerbose')
    params.verbose = false;
  end
  if ~isfield(params, 'optVerbosePerIter')
    params.verbosePerIter = 20;
  end

  % Determine which method to use for optimisation
  switch params.optMethod

    case 'subGradient'
      [optBeta, optStats] = subGradMethod(Ls, Y, lambda, params);

    case 'proxGradient'
      params.useAcceleration = false;
      [optBeta, optStats] = proxGradMethod(Ls, Y, lambda, params);

    case 'proxGradientAccn'
      params.useAcceleration = true;
      [optBeta, optStats] = proxGradMethod(Ls, Y, lambda, params);

    case 'bcdExact'
      [~, optBeta, optStats] = bcd_exact(Y, Ls, lambda, params);

    case 'bcgd'

    case 'admm'

    case 'proxNewton'
    
    otherwise
      error('Unknown Optimisation Method.');

  end

end

