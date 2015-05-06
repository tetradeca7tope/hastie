function [optBeta, optStats] = ...
  addKernelRegOpt(Ls, Y, decomposition, lambda, params)
% Performs Additive Kernel Regression.
% Ks, Y: All Kernels Covariates and Labels
% decomposition is a struct containing the groups.
% lambda: penalty coefficient.
% params: contains ancillary params

% N.B: This function trains using all of X, Y for the given value of lambda. 
% Ideally we should be using CV etc for which we will write a wrapper.

  % prelims
  n = size(Y, 1);

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
      [optBeta, optStats] = bcd_exact(Ls, Y, lambda, params);

    case 'bcgdDiagHessian'
      [optBeta, optStats] = bcgd_ha(Ls, Y, lambda, params);

    case 'admm'
        [optBeta, optStats] = admm(Ls, Y, lambda, params);

    case 'proxNewton'
    
    otherwise
      error('Unknown Optimisation Method.');

  end

end
