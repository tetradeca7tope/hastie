
% This compares between different choices of (d,M);

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/
addpath ../otherMethods/
% addpath ~/libs/libsvm/matlab/  % add libsvm path here
% addpath ~/libs/gpml/, startup;
rng('default');


% dMVals = [3 50; 3 100; 3 200; 5 50; 5 100; 5 200; 8 50; 8 100; 8 200; ...
%   12 50; 12 100; 12 200; 20 50; 20 100; 20 200];
dMVals = [3 50; 3 100; 3 200; 5 50; 5 100; 5 200; 8 50; 8 100; 8 200];
% dMVals = [12 50; 12 100; 12 200; 16 50; 16 100; 16 200; 20 50; 20 100; 20 200];

% Problem Set up
numExperiments = 2;
numDims = 40; nTotal = 600;  maxNumIters = 50;
nCands = (60:60:nTotal)';


% DEBUG
dMVals = [3 50; 3 100; 5 100; 8 30];
numDims = 10; nTotal = 100;  maxNumIters = 10;
nCands = (40:40:nTotal)';

[func, funcProps] = getAdditiveFunction(numDims, 7);
bounds = funcProps.bounds;
nTest = 500;
lambdaRange = [1e-3 1];
numNCands = numel(nCands);

% For the Decomposition
% decomposition.setting = 'espKernel';
decomposition.setting = 'randomGroups';
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

numDMCandidates =  size(dMVals, 1);

% For storing the results
results = cell(numDMCandidates, 1);
for j = 1:numDMCandidates
  results{j} = zeros(numExperiments, numNCands);
end

saveFileName = sprintf('results/v23-%s.mat', datestr(now, 'ddmm-HHMMSS'));


for expIter = 1:numExperiments

  fprintf('Experiment Iter: %d\n', expIter);
  % Generate training data
  Xtr = bsxfun(@plus, ...
    bsxfun(@times, rand(nTotal, numDims), (bounds(:,2) - bounds(:,1))' ), ...
    bounds(:, 1)' );
  Ytr = func(Xtr);

  for nCandIter = 1:numNCands

    n = nCands(nCandIter);
    X = Xtr(1:n, :);
    Y = Ytr(1:n, :);

    for dMCandIter = 1:numDMCandidates

      decomp = decomposition;
      decomp.groupSize = dMVals(dMCandIter, 1);
      decomp.numRandGroups = dMVals(dMCandIter, 2);
      addKernelPredFunc = addKernelRegCV(X, Y, decomp, [], optParams);
      YPred = addKernelPredFunc(Xte);
      currErr = norm(YPred - Yte).^2/nTest;
      results{dMCandIter}(expIter, nCandIter) = currErr;
      fprintf('(d,M) = (%d,%d), Err: %0.4f\n', dMVals(dMCandIter, 1), ...
        dMVals(dMCandIter, 2), currErr);

    end

  end

  save(saveFileName, 'results', 'dMVals', 'numDMCandidates', 'numNCands', ...
    'nCands', 'numExperiments', 'maxNumIters');

end

plotV23Results;

