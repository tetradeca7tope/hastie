% Unit test for addKernelRegOpt
% We will be calling addKernelRegTrainOnly to interface the function.

%clear all;
close all;
%clc;
addpath ../utils/
addpath ../otherMethods/
% rng('default');

% numDims = 20; n = 200; numRandGroups = 200; 
numDims = 40; n = 400; numRandGroups = 200; 
% numDims = 20; n = 12; numRandGroups = 5; % Debug setting
lambda = 1;

% Generate Toy Data
f = @(X) 0.1*(sum(X.^2, 2) + sum(X, 2).^2 + X(:,1) );
Xtr = randn(n, numDims);
Ytr = f(Xtr);
Xte = randn(n, numDims);
Yte = f(Xte);

% Generate the decomposition
maxNumIters = 500;
% decomposition.setting = 'espKernel';
decomposition.setting = 'randomGroups';
% decomposition.setting = 'maxGroupSize';
decomposition.numRandGroups = numRandGroups;
decomposition.maxGroupSize = 3;
decomposition.groupSize = 6;
decomposition.addAll1DComps = false;
%params.optMethod = 'proxGradient';
%params.optMethod = 'proxGradientAccn';
% params.optMethod = 'subGradient';
% params.optMethod = 'bcdExact';
params.optMethod = 'bcgdDiagHessian';
params.maxNumIters = maxNumIters;
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

% Locally Quadratic Regression
lqPred = localPolyKRegressionCV(Xtr, Ytr, [], 2);
Ylq = nwPred(Xte);
lqErr = norm(Ylq - Yte),

% GP
YGP = gpRegWrap(Xtr, Ytr, Xte);
gpErr = norm(YGP - Yte),

% Additive GP
YaddGP = addGPRegWrap(Xtr, Ytr, Xte);
addGPErr = norm(YaddGP - Yte),

