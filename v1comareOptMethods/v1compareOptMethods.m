% This is for comparing the different optimisation methods

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/
rng('default');

compareMethods = {'subGradient', 'subGradientAlpha', 'proxGradient', ...
  'proxGradientAccn', 'bcdExact', 'bcgdDiagHessian', 'admm'};
plotColours = {'b', 'g', 'r', 'k', 'c', 'm', [255 128 0]/255, ...
  [76, 0, 153]/253, [102 102 0]/255, 'y'};
plotMarkers = {'-o', '-+', '-*', '-x', '-s', '-d', '-^', '-p', '->', '-v'};
plotFunc = @loglog;

% Problem Set up
numDims = 20; n = 500; M = 100; maxNumIters = 1000;
% numDims = 6; n = 100; M = 10; maxNumIters = 100;  % For debugging
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = n;
lambda = 0.001;

% For the Decomposition
decomposition.setting = 'randomGroups';
decomposition.numRandGroups = M;
decomposition.groupSize = 4;
% Parameters for Optimisation
optParams.maxNumIters = maxNumIters;
optParams.optVerbose = true;
optParams.tolerance = 0;

% Sample train and test data uniformly within the bounds
Xtr = bsxfun(@plus, ...
  bsxfun(@times, rand(n, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Ytr = func(Xtr);
Yte = func(Xte);

% Store the results here
numMethods = numel(compareMethods);
objHistories = cell(numMethods, 1);
timeHistories = cell(numMethods, 1);

for methIter = 1:numMethods

  optParams.optMethod = compareMethods{methIter};
  [predFunc, optAlpha, optBeta, stats, decomposition] = ...
    addKernelRegTrainOnly(Xtr, Ytr, decomposition, lambda, optParams);
  objHistories{methIter} = stats.objective;
  timeHistories{methIter} = stats.time;

end

% plot results
plotV1Results;


