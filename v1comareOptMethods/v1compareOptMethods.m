% This is for comparing the different optimisation methods

close all;
clear all;
clc;
addpath ../addRKHSRegression/
addpath ../utils/
rng('default');

compareMethods = {'subGradient', 'proxGradient', 'proxGradientAccn', ...
  'bcdExact', 'bcgdDiagHessian'};
plotColours = {'r', 'g', 'b', 'k', 'c'};
plotMarkers = {'-x', '-o', '-*', '-s', '-^'};
plotFunc = @loglog;

% Problem Set up
numDims = 20; n = 500; M = 50; maxNumIters = 2000;
% numDims = 6; n = 100; M = 10; maxNumIters = 100;  % For debugging
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = n;
lambda1 = 1;
lambda2 = 5;

% For the Decomposition
decomposition.setting = 'randomGroups';
decomposition.numRandGroups = M;
decomposition.groupSize = 4;
decomposition.maxGroupSize = 3;
% Parameters for Optimisation
optParams.maxNumIters = maxNumIters;

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
  [predFunc, decomposition, stats] = ...
    addKernelRidgeRegression(Xtr, Ytr, decomposition, lambda1, lambda2, optParams);
  objHistories{methIter} = stats.objective;
  timeHistories{methIter} = stats.time;

end

% Now plot the results out
minObjs = zeros(numMethods, 1);
maxObjs = zeros(numMethods, 1);
maxTimes = zeros(numMethods, 1);

figure;
for j = 1:numMethods
  numCurrIters = size(objHistories{j}, 1);
  plotFunc(timeHistories{j}, objHistories{j}, plotMarkers{j}, ...
    'Color', plotColours{j});
  hold on,

  % some book keeping
  minObjs(j) = objHistories{j}(end);
  maxObjs(j) = objHistories{j}(1);
  maxTimes(j) = timeHistories{j}(end);
end
minObjVal = min(minObjs);
maxObjVal = max(maxObjs);
maxTimeVal = max(maxTimes);
objDiff = maxObjVal - minObjVal;
xlim([0 maxTimeVal*1.05]);
ylim([minObjVal maxObjVal]);
legend(compareMethods);
title('Objective vs Time (seconds)');

figure;
for j = 1:numMethods
  numCurrIters = size(objHistories{j}, 1);
  plotFunc(1:numCurrIters, objHistories{j}, plotMarkers{j}, ...
    'Color', plotColours{j});
  hold on,

  % some book keeping
end
xlim([0 optParams.maxNumIters*1.1]);
ylim([minObjVal maxObjVal]);
legend(compareMethods);
title('Objective vs Iteration');




