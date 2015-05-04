% Unit test for addKernelRegOpt
% We will be calling addKernelRegTrainOnly to interface the function.

clear all;
close all;
clc;
addpath ../utils/
addpath ../otherMethods/
rng('default');

numDims = 20; n = 400; numRandGroups = 200; 

% % Generate Toy Data
% f = @(X) 0.1*(sum(X.^2, 2) + sum(X, 2).^2 + X(:,1) );
% Xtr = randn(n, numDims);
% Ytr = f(Xtr);
% Xte = randn(n, numDims);
% Yte = f(Xte);

% Set up
% [func, funcProps] = getAdditiveFunction(numDims, numDims);
[func, funcProps] = getAdditiveFunction(numDims, round(numDims/2) -1);
bounds = funcProps.bounds;
nTest = n;

% Sample train and test data uniformly within the bounds
Xtr = bsxfun(@plus, ...
  bsxfun(@times, rand(n, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Ytr = func(Xtr);
Yte = func(Xte);


% Generate the decomposition
% decomposition.setting = 'espKernel';
decomposition.setting = 'randomGroups';
% decomposition.setting = 'maxGroupSize';
decomposition.numRandGroups = numRandGroups;
decomposition.groupSize = 8;
decomposition.addAll1DComps = false;
lambdaRange = [1e-16 .1];
% params.optMethod = 'proxGradient';
% params.optMethod = 'proxGradientAccn';
% params.optMethod = 'subGradient';
% params.optMethod = 'bcdExact';
params.optMethod = 'bcgdDiagHessian';
params.maxNumIters = 200;
params.optVerbose = true;
params.numLambdaCands = 20;
[predFunc, optAlpha, optBeta, decomposition, bestLambda, optStats] = ...
  addKernelRegCV(Xtr, Ytr, decomposition, lambdaRange, params);

% Now do prediction
Ypred = predFunc(Xte);
addErr = norm(Ypred - Yte),

% Nadaraya Watson Regression
nwPred = localPolyKRegressionCV(Xtr, Ytr, [], 0);
YNW = nwPred(Xte);
nwErr = norm(YNW - Yte),

% Locally Quadratic Regression
lqPred = localPolyKRegressionCV(Xtr, Ytr, [], 1);
Ylq = lqPred(Xte);
lqErr = norm(Ylq - Yte),

% Additive GP
YaddGP = addGPRegWrap(Xtr, Ytr, Xte);
addGPErr = norm(YaddGP - Yte),

