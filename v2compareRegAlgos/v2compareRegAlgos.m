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

% Problem Set up
numExperiments = 2;
numDims = 20; nTotal = 500; M = 50; maxNumIters = 2000;
% numDims = 6; nTotal = 150; M = 10; maxNumIters = 100;  % For debugging
nCands = (60:60:nTotal)';
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = 500;
lambda1 = 1;
lambda2 = 5;
numCandidates = numel(nCands);

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

% Sample test data
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Yte = func(Xte);

% Store the results here
numRegAlgos = numel(regressionAlgorithms);

% For storing the results
results = cell(numRegAlgos, 1);
for j = 1:numRegAlgos
  results{j} = zeros(numExperiments, numCandidates);
end


for expIter = 1:numExperiments

  fprintf('Experiment Iter: %d\n', expIter);
  % Generate training data
  Xtr = bsxfun(@plus, ...
    bsxfun(@times, rand(nTotal, numDims), (bounds(:,2) - bounds(:,1))' ), ...
    bounds(:, 1)' );
  Ytr = func(Xtr);

  for candIter = 1:numCandidates

    n = nCands(candIter);
    X = Xtr(1:n, :);
    Y = Ytr(1:n, :);

    % Method 1: add-KR
    [addKRPredFunc] = addKernelRidgeRegression(X, Y, decomposition, ...
      lambda1, lambda2, optParams);
    YPred = addKRPredFunc(Xte);
    results{1}(expIter, candIter) = norm(YPred - Yte);
    
    % Method 2: KR
    [krPredFunc] = addKernelRidgeRegression(X, Y, krDecomposition, ...
      lambda1, lambda2, optParams);
    YPred = krPredFunc(Xte);
    results{2}(expIter, candIter) = norm(YPred - Yte);

    % Method 3: NW
    YPred = localPolyKRegressionCV(Xte, X, Y, [], 0);
    results{3}(expIter, candIter) = norm(YPred - Yte);
    

  end

end

plotV2Results;

