function [predFunc, decomposition] = addKernelRidgeRegression(X, Y, ...
  decomposition, lambda1, lambda2, params)
% Performs Additive RKHS Regression (w/o the cross validation).
% X, Y: Covariates and Labels
% decomposition is a struct containing the groups.
% lambda1, lambda2: The penalty coefficients in the objective.

  % prelims
  n = size(X, 1);

  % Obtain the Kernel(s)
  [kernelFunc, decomposition] = kernelSetup(X, Y, decomposition);
  groups = decomposition.groups;
  M = numel(groups);
  [~, allKs] = kernelFunc(X, X); % compute the kernel

  % Determine which method to use for optimisation
  switch params.optMethod

    case 'proxGrad'
      fprintf('Using Proximal Gradient Descent.\n');
      optAlpha = addKernelRidgeProxGradDesc(allKs, Y, lambda1, lambda2, params);

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
    Ypred = allKs(:,:,j) * optAlpha(:,j);
  end

end

