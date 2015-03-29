% Unit test for Additive Kernel Ridge Regression

clear all;
close all;
clc;
addpath ~/libs/kky-matlab/utils/
addpath ../utils/
rng('default');

numDims = 10;
n = 200;
lambda1 = 1;
lambda2 = 5;

% Generate Toy Data
f = @(X) 0.1*(sum(X.^2, 2) + sum(X, 2).^2 + X(:,1) );
Xtr = randn(n, numDims);
Ytr = f(Xtr);
Xte = randn(n, numDims);
Yte = f(Xte);

% Generate the decomposition
decomposition.setting = 'randomGroups';
% decomposition.setting = 'maxGroupSize';
decomposition.numRandGroups = 20;
decomposition.maxGroupSize = 3;
decomposition.groupSize = 4;
% params.optMethod = 'proxGradient';
% params.optMethod = 'subGradient';
% params.optMethod = 'bcdExact';
params.optMethod = 'bcgdDiagHessian';
params.maxNumIters = 1000;
[predFunc, decomposition, stats] = ...
  addKernelRidgeRegression(Xtr, Ytr, decomposition, lambda1, lambda2, params);

% Now do a prediction
Ypred = predFunc(Xte);
addErr = norm(Ypred - Yte),

% Nadaraya Watson Regression
YNW = localPolyKRegressionCV(Xte, Xtr, Ytr, [], 0);
nwErr = norm(YNW - Yte),

