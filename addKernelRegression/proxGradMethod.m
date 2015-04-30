function [optBeta, optStats] = proxGradMethod(Ls, Y, lambda, params)
% Implements proximal Gradient Method to optimise the objective

% Ls: The cholesky decomposition of each kernel matrix
% Y: The labels:
% lambda: The coefficients for the regularization penalty.

  % Optimisation Parameters
  dampFactor = 0.9; % for backtracking line search

  % prelims
  n = size(Y, 1);
  M = size(Ls, 3);
  % Create function handle for the objective
  objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
  smoothObjGrad = @(arg) computeSmoothObjGradBeta(arg, Ls, Y);
  % Check for optimisation parameters
  

end
