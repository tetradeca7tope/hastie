% Unit test for addKernelRegOpt
% We will be calling addKernelRegTrainOnly to interface the function.

clear all;
close all;
clc;
addpath ../utils/
addpath ../otherMethods/
rng('default');

numDims = 20; n = 400; numRandGroups = 200; 

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
decomposition.groupSize = 6;
decomposition.addAll1DComps = false;
lambdaRange = [1e-12 1];
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
nwPred = localPolyKRegressionCV(Xtr, Ytr, [], 0);
YNW = nwPred(Xte);
nwErr = norm(YNW - Yte),

