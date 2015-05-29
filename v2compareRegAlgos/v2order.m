% This script is for comparing different values for the Order

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/

numDims = 12;
orderCands = 1:10;
% orderCands = [1 2 4 6 10];
numOrderCands = numel(orderCands);

% Problem Set up
numExperiments = 5;
nTotal = 8000; nCands = round(logspace(log10(50), log10(nTotal), 15));
% nTotal = 300; nCands = (60:60:nTotal)'; % DEBUG
numNCandidates = numel(nCands);

numGroupDims = numDims;
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = 2000;

% Sample test data
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Yte = func(Xte);

% Book Keeping 
saveFileName = sprintf('results/v2order-%s.mat', datestr(now, 'mmdd-HHMMSS'));
% Store the results here
results = cell(numOrderCands, 1);
for j = 1:numOrderCands
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

    for orderIter = 1:numOrderCands
      params.orderCands = orderCands(orderIter);
      predFunc = addKRR(X, Y, params);
      YPred = predFunc(Xte);
      currErr = norm(YPred - Yte).^2/nTest;
      results{orderIter}(expIter, candIter) = currErr;
      fprintf('Order %d: Err: %.5f\n\n', orderCands(orderIter), currErr);
    end

  end

  % Save results
  save(saveFileName, 'numOrderCands', 'orderCands', 'results', ...
    'numNCandidates', 'nCands', 'numExperiments');

end

plotV2OrderResults;
