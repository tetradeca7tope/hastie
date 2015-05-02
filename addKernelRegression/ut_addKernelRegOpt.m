% Unit test for addKernelRegOpt
% We will be calling addKernelRegTrainOnly to interface the function.

%clear all;
close all;
%clc;
addpath ../utils/
addpath ../otherMethods/
rng('default');

numDims = 10; n = 200; numRandGroups = 200; 
% numDims = 10; n = 12; numRandGroups = 5; % Debug setting
lambda = 1;

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
%params.optMethod = 'proxGradient';
%params.optMethod = 'proxGradientAccn';
% params.optMethod = 'subGradient';
% params.optMethod = 'bcdExact';
params.optMethod = 'bcgdDiagHessian';
params.maxNumIters = 1000;
params.optVerbose = true;
[predFunc, optAlpha, optBeta, optStats, decomposition] = ...
  addKernelRegTrainOnly(Xtr, Ytr, decomposition, lambda, params);

% Now do a prediction
Ypred = predFunc(Xte);
addErr = norm(Ypred - Yte),

% Nadaraya Watson Regression
nwPred = localPolyKRegressionCV(Xtr, Ytr, [], 0);
YNW = nwPred(Xte);
nwErr = norm(YNW - Yte),
