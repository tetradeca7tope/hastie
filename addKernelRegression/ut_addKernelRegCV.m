% Unit test for addKernelRegOpt
% We will be calling addKernelRegTrainOnly to interface the function.

clear all;
close all;
clc;
addpath ../utils/
addpath ../otherMethods/
rng('default');

numDims = 10; n = 400; numRandGroups = 30; 

% Generate Toy Data
f = @(X) 0.1*(sum(X.^2, 2) + sum(X, 2).^2 + X(:,1) );
Xtr = randn(n, numDims);
Ytr = f(Xtr);
Xte = randn(n, numDims);
Yte = f(Xte);

% Generate the decomposition
decomposition.setting = 'randomGroups';
% decomposition.setting = 'maxGroupSize';
decomposition.numRandGroups = numRandGroups;
decomposition.maxGroupSize = 3;
decomposition.groupSize = 4;
decomposition.addAll1DComps = false;
lambdaRange = [1e-8 1];
% params.optMethod = 'proxGradient';
% params.optMethod = 'proxGradientAccn';
% params.optMethod = 'subGradient';
% params.optMethod = 'bcdExact';
params.optMethod = 'bcgdDiagHessian';
params.maxNumIters = 500;
params.optVerbose = true;
params.numLambdaCands = 20;
[predFunc, optAlpha, optBeta, decomposition, bestLambda, optStats] = ...
  addKernelRegCV(Xtr, Ytr, decomposition, lambdaRange, params);

% Now do a prediction
Ypred = predFunc(Xte);
addErr = norm(Ypred - Yte),

% Nadaraya Watson Regression
YNW = localPolyKRegressionCV(Xte, Xtr, Ytr, [], 0);
nwErr = norm(YNW - Yte),

