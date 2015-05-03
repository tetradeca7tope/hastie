% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addRKHSRegression/
addpath ../utils/
addpath ~/libs/kky-matlab/utils/
rng('default');

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
lambdaRange = [1e-7 100];

% Save file name
saveFileName = sprintf('results/%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS') );

regressionAlgorithms = ...
  {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'spam'};
regressionAlgorithms = { ...
  {'KRR',      @(X,Y,Xte) kernelRidgeReg(X, Y, struct())}, ...
  {'NW',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],0)}, ...
  {'LL',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],1)}, ...
  {'LQ',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],2)}, ...
  {'GP',       @(X,Y,Xte) gpRegWrap(X,Y,Xte)}, ...
  {'addGP',    @(X,Y,Xte) addGPWrap(X,Y,Xte)}, ...

numRegAlgos = numel(regressionAlgorithms);


results = zeros(numRegAlgos, 1);

% Now run each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnt = 0;

% Method 1: add-KR
cnt = cnt + 1;
predFunc = addKernelRegCV(Xtr, Ytr, decomposition, lambdaRange, optParams);
YPred = predFunc(Xte);
predError = norm(YPred-Yte).^2/nTe;
results(cnt) = predError;
fprintf('Method: %s, err: %.4f\n', regressionAlgorithms{cnt}, predError);

% Method 2: KR
cnt = cnt + 1;
predFunc = kernelRidgeReg(Xtr, Ytr, struct());
YPred = predFunc(Xte);
predError = norm(YPred-Yte).^2/nTe;
results(cnt) = predError;
fprintf('Method: %s, err: %.4f\n', regressionAlgorithms{cnt}, predError);

% Method 2: KR
cnt = cnt + 1;
predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 0);
YPred = predFunc(Xte);
predError = norm(YPred-Yte).^2/nTe;
results(cnt) = predError;
fprintf('Method: %s, err: %.4f\n', regressionAlgorithms{cnt}, predError);

results
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


