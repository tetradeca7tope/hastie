% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addRKHSRegression/
addpath ../utils/
addpath ~/libs/kky-matlab/utils/
rng('default');

regressionAlgorithms = {'add-KR', 'KR', 'NW'};

% Load the dataset
Data = load('train_data.txt'); N = size(Data, 1);
Data = Data(randperm(N), 2:27);
attrs = [1:14, 20:26]; label = 15;
n = round(N/2);
trainIdxs = 1:n; testIdxs = (n+1):N;
Xtr = Data(trainIdxs, attrs);
Ytr = Data(trainIdxs, label);
Xte = Data(testIdxs, attrs);
Yte = Data(testIdxs, label);
numDims = size(Xtr, 2);

% Problem Set up
M = 50; maxNumIters = 2000;
lambda1 = 1;
lambda2 = 5;
% For the Decomposition
decomposition.setting = 'randomGroups';
decomposition.numRandGroups = M;
decomposition.groupSize = 4;
decomposition.maxGroupSize = 3;
% Parameters for Optimisation
optParams.maxNumIters = maxNumIters;
optParams.optMethod = 'proxGradientAccn';
% Decomposition for plain Kernel Ridge Regression
krDecomposition.setting = 'groups';
krDecomposition.groups = { 1:numDims };

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

