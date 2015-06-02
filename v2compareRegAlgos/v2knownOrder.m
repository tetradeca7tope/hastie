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
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'SpAM'};
% regressionAlgorithms = ...
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN', 'SpAM'};
regressionAlgorithms = ...
  {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN'};
% regressionAlgorithms = ...
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'kNN'};

% Problem Set up
numExperiments = 5; nTotal = 2000;
numDims = 15; order = 3; nCands = (120:120:nTotal)';
addKrrParams.orderCands = [order];

[func, funcProps] = getOrderAddFunction(numDims, order);
bounds = funcProps.bounds;
nTest = 2000;

% Sample test data
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Yte = func(Xte);

% Book Keeping 
numNCandidates = numel(nCands);
saveFileName = sprintf('results/v2known-%s.mat', datestr(now, 'mmdd-HHMMSS'));
% Store the results here
numRegAlgos = numel(regressionAlgorithms);
results = cell(numRegAlgos, 1);
for j = 1:numRegAlgos
  results{j} = zeros(numExperiments, numNCandidates);
end

% Start Experiments here
for expIter = 1:numExperiments

  fprintf('\nExperiment Iter: %d/%d\n================================\n', ...
    expIter, numExperiments);
  % Generate training data
  Xtr = bsxfun(@plus, ...
    bsxfun(@times, rand(nTotal, numDims), (bounds(:,2) - bounds(:,1))' ), ...
    bounds(:, 1)' );
  Ytr = func(Xtr);

  for candIter = 1:numNCandidates

    n = nCands(candIter);
    X = Xtr(1:n, :);
    Y = Ytr(1:n, :);
    fprintf('n = %d\n------------------------\n', n);

    for algoIter = 1:numRegAlgos

      switch regressionAlgorithms{algoIter}

        case 'addKRR' 
          predFunc = addKRR(X, Y, addKrrParams);
          YPred = predFunc(Xte);

        case 'KRR'
          predFunc = kernelRidgeReg(X, Y, struct());
          YPred = predFunc(Xte);

        case 'NW'
          predFunc = localPolyRegressionCV(X, Y, [], 0);
          YPred = predFunc(Xte);

        case 'LL'
          predFunc = localPolyRegressionCV(X, Y, [], 1);
          YPred = predFunc(Xte);

        case 'LQ'
          predFunc = localPolyRegressionCV(X, Y, [], 2);
          YPred = predFunc(Xte);

        case 'GP'
          YPred = gpRegWrap(X, Y, Xte);

        case 'addGP'
          YPred = addGPRegWrap(X, Y, Xte);

        case 'SVR'
          predFunc = svmRegWrap(X, Y, 'eps');
          YPred = predFunc(Xte);

        case 'kNN'
          predFunc = KnnRegressionCV(X, Y);
          YPred = predFunc(Xte);

        case 'SpAM'
          predFunc = SpamRegressionCV(Xtr, Ytr, []);
          YPred = predFunc(Xte);

        otherwise
          errorStr = sprintf('Unknown Method %s\n', regressionAlgorithms{algoIter});
          error(errorStr);

      end

      % Now record errors
      currErr = norm(YPred - Yte).^2/nTest;
      results{algoIter}(expIter, candIter) = currErr;
      fprintf('%s: Err: %.5f\n\n', regressionAlgorithms{algoIter}, currErr);

    end

  end

  % Save results
  save(saveFileName, 'numDims', 'order', 'results', 'numRegAlgos', ...
    'numNCandidates', 'regressionAlgorithms', 'nCands', 'numExperiments');

end

plotV2Results;

