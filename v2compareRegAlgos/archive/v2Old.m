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

regressionAlgorithms = ...
  {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'spam'};
% regressionAlgorithms = ...
%   {'add-KR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN', 'spam'};
% regressionAlgorithms = ...
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'kNN'};

% Problem Set up
numExperiments = 2;
% numDims = 20; nTotal = 1100; nCands = (120:120:nTotal)';
numDims = 6; nTotal = 300; nCands = (60:60:nTotal)'; % DEBUG

[func, funcProps] = getAdditiveFunction(numDims, round(numDims/2)-1);
bounds = funcProps.bounds;
nTest = 2000;

% For the Decomposition
decomposition.setting = 'espKernel';
optParams.maxNumIters = maxNumIters;
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

% SAve results
saveFileName = sprintf('results/v2-%s.mat', datestr(now, 'mmdd-HHMMSS'));

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

    for algoIter = 1:numRegAlgos

      switch regressionAlgorithms{algoIter}

        case 'addKRR' 
          predFunc = addKRR(X, Y);
          YPred = predFunc(Xte);

        case 'KRR'
          predFunc = kernelRidgeReg(X, Y, struct());
          YPred = predFunc(Xte);

        case 'NW'
          predFunc = localPolyKRegression(X, Y, [], 0);
          YPred = predFunc(Xte);

        case 'LL'
          predFunc = localPolyKRegression(X, Y, [], 1);
          YPred = predFunc(Xte);

        case 'LQ'
          predFunc = localPolyKRegression(X, Y, [], 2);
          YPred = predFunc(Xte);

        case 'GP'
          YPred = gpRegWrap(X, Y, Xte);

        case 'addGP'
          YPred = addGPRegWrap(X, Y, Xte);

        case 'SVR'
          predFunc = svmRegWrap(X, Y, 'eps');
          YPred = predFunc(Xte);

        case 'KNN'
          predFunc = KnnRegressionCV(X, Y);
          YPred = predFunc(Xte);

        case 'SpAM'
          predFunc = SpamRegressionCV(Xtr, Ytr, []);
          YPred = spamPredFunc(Xte);

      end

      % Now record errors
      currErr = norm(YPred - Yte).^2/nTest;
      results{algoIter}(expIter, candIter) = currErr;
      fprintf('%s: Err: %.5f\n', regressionAlgorithms{algoIter}, currErr);

    end

  end

  % Save results
  save(saveFileName, 'results', 'numRegAlgos', 'numCandidates', ...
    'regressionAlgorithms', 'nCands', 'numExperiments');

end

plotV2Results;

