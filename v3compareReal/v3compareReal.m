% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addRKHSRegression/
addpath ../utils/
addpath ~/libs/kky-matlab/utils/
rng('default');

regressionAlgorithms = ...
  {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'spam'};

% Determine dataset
dataset = 'parkinson21';

% Load data
[Xtr, Ytr, Xte, Yte] = genDataset(dataset);
[nTr, numDims] = size(Xtr);
nTe = size(Xte, 1);

% For the Decomposition
decomposition.numRandGroups = 200;
decomposition.groupSize = 5;
decomposition.setting = 'espKernel';
% decomposition.setting = 'randomGroups';

% Parameters for Optimisation
optParams.maxNumIters = maxNumIters;
% optParams.optMethod = 'proxGradientAccn';
optParams.optMethod = 'bcgdDiagHessian';
% Decomposition for plain Kernel Ridge Regression
krDecomposition.setting = 'groups';
krDecomposition.groups = { 1:numDims };

% Save file name
saveFileName = sprintf('results/%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS') );

% Now run each method

% Method 1: add-KR
[addKRPredFunc] = addKernelRidgeRegression(Xtr, Ytr, decomposition, ...
  lambda1, lambda2, optParams);
YPred = addKRPredFunc(Xte);
fprintf('addKR: %0.4f\n', norm(YPred-Yte));

% Method 2: KR
[krPredFunc] = addKernelRidgeRegression(Xtr, Ytr, krDecomposition, ...
  lambda1, lambda2, optParams);
YPred = krPredFunc(Xte);
fprintf('KR: %0.4f\n', norm(YPred-Yte));

% Method 3: NW
YPred = localPolyKRegressionCV(Xte, Xtr, Ytr, [], 0);
fprintf('NW: %0.4f\n', norm(YPred-Yte));

