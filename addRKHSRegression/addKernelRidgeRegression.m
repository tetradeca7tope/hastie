function [predFunc, decomposition, optStats] = ...
  addKernelRidgeRegression(X, Y, decomposition, lambda1, lambda2, params)
% Performs Additive RKHS Regression (w/o the cross validation).
% X, Y: Covariates and Labels
% decomposition is a struct containing the groups.
% lambda1, lambda2: The penalty coefficients in the objective.

  lambda2Prime = 0.01; % Calvin's thing

  % prelims
  n = size(X, 1);

  % Obtain the Kernel(s)
  [kernelFunc, decomposition] = kernelSetup(X, Y, decomposition);
  groups = decomposition.groups;
  M = numel(groups);
  [~, allKs] = kernelFunc(X, X); % compute the kernel

  % Determine which method to use for optimisation
  fprintf('\nAdditive Kernel Ridge Regresion: \n');
  fprintf('n = %d, M = %d \n', n, M);
  switch params.optMethod

    case 'subGradient'
      fprintf('Using SubGradient Method.\n');
      [optAlpha, optStats] = ...
        subgradient(Y, allKs, lambda1, lambda2Prime, lambda2, params);

    case 'proxGradient'
      fprintf('Using Proximal Gradient Descent.\n');
      params.useAcceleration = false;
      [optAlpha, optStats] = ...
        proxGradMethod(allKs, Y, lambda1, lambda2, params);

    case 'proxGradientAccn'
      fprintf('Using Proximal Gradient Descent with Acceleration.\n');
      params.useAcceleration = true;
      [optAlpha, optStats] = ...
        proxGradMethod(allKs, Y, lambda1, lambda2, params);
      
    case 'bcdExact'
      fprintf('Using Block Coordinate Descent - Exact.\n');
      [optAlpha, optStats] = ...
        bcd_exact(Y, allKs, lambda1, lambda2Prime, lambda2, params);

    case 'bcgdDiagHessian'
      fprintf('Using BCGD with Diagoanal Hessian Approximation.\n');
      [optAlpha, optStats] = ...
        bcgd_ha(Y, allKs, lambda1, lambda2Prime, lambda2, params);

    otherwise
      error('Unknown Optimisation Method.');

  end


  % Now create a function handle to obtain the predictions
  predFunc = @(Xtest) prediction(Xtest, X, groups, optAlpha, kernelFunc);

end


% A predictor function
function Ypred = prediction(Xte, X, groups, optAlpha, kernelFunc)

  M = numel(groups);
  numTest = size(Xte, 1);
  Ypred = zeros(numTest, 1);

  [~, allKs] = kernelFunc(Xte, X);
  for j = 1:M
    Ypred = Ypred + allKs(:,:,j) * optAlpha(:,j);
  end

end

