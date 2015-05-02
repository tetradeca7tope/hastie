% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/
addpath ../otherMethods/
addpath ~/libs/libsvm/matlab/  % add libsvm path here
addpath ~/libs/gpml/, startup;
rng('default');

regressionAlgorithms = {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP'};

% Problem Set up
numExperiments = 2;
numDims = 20; nTotal = 500; M = 50; maxNumIters = 50;
% numDims = 6; nTotal = 150; M = 10; maxNumIters = 20;  % For debugging
nCands = (60:60:nTotal)';
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = 500;
lambdaRange = [1e-3 1];
numCandidates = numel(nCands);

% For the Decomposition
decomposition.setting = 'randomGroups';
decomposition.numRandGroups = M;
decomposition.groupSize = 5;
% Parameters for Optimisation
optParams.maxNumIters = maxNumIters;
% optParams.optMethod = 'proxGradientAccn';
optParams.optMethod = 'bcgdDiagHessian';

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
    addKernelPredFunc = addKernelRegCV(X, Y, decomposition, [], optParams);
    YPred = addKernelPredFunc(Xte);
    results{1}(expIter, candIter) = norm(YPred - Yte);
    
    % Method 2: KR
    [krPredFunc] = kernelRidgeReg(X, Y, struct());
    YPred = krPredFunc(Xte);
    results{2}(expIter, candIter) = norm(YPred - Yte);

    % Method 3: NW
    nwPredFunc = localPolyKRegressionCV(X, Y, [], 0);
    YPred = nwPredFunc(Xte);
    results{3}(expIter, candIter) = norm(YPred - Yte);
    
    % Method 4: Locally Linear
    llPredFunc = localPolyKRegressionCV(X, Y, [], 1);
    YPred = llPredFunc(Xte);
    results{4}(expIter, candIter) = norm(YPred - Yte);

    % Method 5: Locally Quadratic 
    lqPredFunc = localPolyKRegressionCV(X, Y, [], 2);
    YPred = lqPredFunc(Xte);
    results{5}(expIter, candIter) = norm(YPred - Yte);

    % Method 6: GP
    YPred = gpRegWrap(X, Y, Xte);
    results{6}(expIter, candIter) = norm(YPred - Yte);
 
%     % Method 7: additive-GP
%     Ypred = addGPRegWrap(X, Y, Xte);
%     results{7}(expIter, candIter) = norm(YPred - Yte);

  end

end

plotV2Results;

