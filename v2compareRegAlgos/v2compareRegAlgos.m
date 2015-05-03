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

% regressionAlgorithms = ...
%   {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'spam'};
% regressionAlgorithms = ...
%   {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN', 'spam'};
regressionAlgorithms = ...
  {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN'};

% Problem Set up
numExperiments = 2;
% numDims = 20; nTotal = 500; M = 200; maxNumIters = 300;
numDims = 20; nTotal = 1100; M = 200; maxNumIters = 400;
% numDims = 6; nTotal = 300; M = 10; maxNumIters = 20;  % For debugging
% nCands = (60:60:nTotal)';
nCands = (120:120:nTotal)';
[func, funcProps] = getAdditiveFunction(numDims, round(numDims/2)-1);
bounds = funcProps.bounds;
nTest = 500;
lambdaRange = [1e-3 1];
numCandidates = numel(nCands);

% For the Decomposition
decomposition.setting = 'espKernel';
% decomposition.setting = 'randomGroups';
% decomposition.numRandGroups = M;
% decomposition.groupSize = 8;

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

    cnt = 0;
    % Method 1: add-KR
    cnt = cnt + 1;
    addKernelPredFunc = addKernelRegCV(X, Y, decomposition, [], optParams);
    YPred = addKernelPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));
    
    % Method 2: KR
    cnt = cnt + 1;
    [krPredFunc] = kernelRidgeReg(X, Y, struct());
    YPred = krPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));

    % Method 3: NW
    cnt = cnt + 1;
    nwPredFunc = localPolyKRegressionCV(X, Y, [], 0);
    YPred = nwPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));
    
    % Method 4: Locally Linear
    cnt = cnt + 1;
    llPredFunc = localPolyKRegressionCV(X, Y, [], 1);
    YPred = llPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));
    
    % Method 4: Locally Quadratic
    cnt = cnt + 1;
    lqPredFunc = localPolyKRegressionCV(X, Y, [], 2);
    YPred = lqPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));

    % Method 6: GP
    cnt = cnt + 1;
    YPred = gpRegWrap(X, Y, Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));
 
%     % Method 7: additive-GP
%     cnt = cnt + 1;
%     YPred = addGPRegWrap(X, Y, Xte);
%     results{cnt}(expIter, candIter) = norm(YPred - Yte);
%     fprintf('Method: %s, Err: %0.4f\n', ...
%       regressionAlgorithms{cnt}, norm(YPred-Yte));
 
    % Method 8: SVR 
    cnt = cnt + 1;
    svPredFunc = svmRegWrap(Xtr, Ytr, 'eps');
    YPred = svPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));

    % Method 9: KNN
    cnt = cnt + 1;
    kNNPredFunc = KnnRegressionCV(Xtr, Ytr);
    [~, YPred] = kNNPredFunc(Xte);
    results{cnt}(expIter, candIter) = norm(YPred - Yte);
    fprintf('Method: %s, Err: %0.4f\n', ...
      regressionAlgorithms{cnt}, norm(YPred-Yte));
 
%     % Method 10: spam
%     cnt = cnt + 1;
%     spamPredFunc = SpamRegressionCV(Xtr, Ytr, []);
%     [~, YPred] = spamPredFunc(Xte);
%     results{cnt}(expIter, candIter) = norm(YPred - Yte);
%     fprintf('Method: %s, Err: %0.4f\n', ...
%       regressionAlgorithms{cnt}, norm(YPred-Yte));

  end

end

plotV2Results;

