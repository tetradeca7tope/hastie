% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/
addpath ../otherMethods/
addpath ../otherMethods/cosso/
addpath ../otherMethods/ARESLab/
addpath ../otherMethods/RBF/
addpath ../otherMethods/M5PrimeLab/
addpath ../otherMethods/sqb-0.1/build/
addpath ~/libs/libsvm/matlab/
addpath ~/libs/gpml/, startup;
rng('default');

% regressionAlgorithms = ...
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'addGP', 'SVR', 'kNN', 'SpAM'};
% regressionAlgorithms = ...
%   {'addKRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN', 'SpAM'};
% regressionAlgorithms = ...
%   {'Add-KRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'SVR', 'kNN'};
% regressionAlgorithms = ...
%   {'Add-KRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'RT'};
regressionAlgorithms = ...
  {'Add-KRR', 'KRR', 'NW', 'LL', 'LQ', 'GP', 'kNN'};

% Problem Set up
numExperiments = 10;
% numDims = 20; nTotal = 1100; nCands = (120:120:nTotal)';
% numDims = 100; nTotal = 1100; nCands = (120:120:nTotal)';
% numDims = 25; nTotal = 1100; nCands = (120:120:nTotal)';

% numGroupDims = 5;
% numGroupDims = 10;
% numGroupDims = round(numDims/2) - 1;
% numGroupDims = numDims;


% Experiments
% ===========================
nTotal = 2000; nCands = (120:120:nTotal)'; 
% numDims = 20; numGroupDims = 9;
% numDims = 30; numGroupDims = 5;
numDims = 30; numGroupDims = 9;
% numDims = 50; numGroupDims = 50;
% numDims = 40; numGroupDims = 20;
% numDims = 24; numGroupDims = 7;
% numDims = 25; numGroupDims = 8;
% numDims = 60; numGroupDims = 13;
% numDims = 80; numGroupDims = 30;
% numDims = 100; numGroupDims = 30;
% numDims = 100; numGroupDims = 5;

% DEBUG
% numExperiments = 2; numDims = 6; numGroupDims = 3; nTotal = 300;
% nCands = (60:60:nTotal)'; % DEBUG

[func, funcProps] = getAdditiveFunction(numDims, numGroupDims);
bounds = funcProps.bounds;
nTest = 2000;

% Sample test data
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Yte = func(Xte);

% Book Keeping 
numNCandidates = numel(nCands);
saveFileName = sprintf('results/v2new-%s.mat', datestr(now, 'mmdd-HHMMSS'));
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
    fprintf('D = %d, n = %d\n------------------------\n', numDims, n);

    for algoIter = 1:numRegAlgos

      switch regressionAlgorithms{algoIter}

        case 'Add-KRR' 
          predFunc = addKRR(X, Y);
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

        case 'RT'
          YPred = regTree(Xtr, Ytr, Xte);

        case 'BT'
          YPred = boostTreeWrap(Xtr, Ytr, Xte);

        case 'BF'
          predFunc = backFitting(Xtr, Ytr);
          YPred = predFunc(Xte);

        case 'RBFI'
          YPred = rbfInterpol(Xtr, Ytr, Xte);

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
  save(saveFileName, 'numDims', 'numGroupDims', 'results', 'numRegAlgos', ...
    'numNCandidates', 'regressionAlgorithms', 'nCands', 'numExperiments');

end

plotV2Results;

