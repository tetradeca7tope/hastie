function [predFunc, optAlpha, optBeta, decomposition, bestLambda, optStats] =...
  addKernelRegCV(X, Y, decomposition, lambdaRange, params)
% Performs Additive RKHS regression and selects the optimal hyper parameters via
% cross validation.
% X, Y: Covariates and Labels
% decomposition is a struct containing the groups
% lambdaRange: range of values for regularization penalty

  % prelims
  n = size(X,1);
  % shuffle the data
  shuffleOrder = randperm(n);
  X = X(shuffleOrder, :);
  Y = Y(shuffleOrder, :);

  % Params for cross validation
  if ~isfield(params, 'numPartsKFoldCV')
    params.numPartsKFoldCV = 5;
  end
  if ~isfield(params, 'numTrialsCV')
    params.numTrialsCV = 2;
  end
  if ~isfield(params, 'numLambdaCands')
    params.numLambdaCands = 20;
  end
  % Copy over to workspace
  numPartsKFoldCV = params.numPartsKFoldCV;
  numLambdaCands = params.numLambdaCands;

  % Obtain the kernel Function and the decomposition
  [kernelFunc, decomposition] = kernelSetup(X, Y, decomposition);
  decomp = decomposition;
  M = decomp.M;
  decomp.setting = 'groups';

  % Set things up for cross validation
  if isempty(lambdaRange), lambdaRange = [1e-7 100]; end
  lambdaCands = fliplr( ...
    logspace(log10(lambdaRange(1)), log10(lambdaRange(2)), numLambdaCands) )';
  errorAccum = zeros(numLambdaCands, 1);
  normBetaVals = zeros(numLambdaCands, M);

  %% Cross validation begins here
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for cvIter = 1:params.numTrialsCV

    % So here's the plan in this loop.
    % 1. First we will split the data into a training and validation set.
    % 2. We will obtain the Kernels/ Cholesky Decomposition and then optimise
    %    using the trainig set.
    % 3. The test set error will accumulated in errorAccum
    % 4. We will initialise beta to zero in the first iteration but from thereon
    %    we will use soft starts.

    % 1. Obtain Train and Test Sets
    testStartIdx = round( (cvIter-1)*n/numPartsKFoldCV + 1);
    testEndIdx = round( cvIter*n/numPartsKFoldCV );
    trainIdxs = [1:(testStartIdx-1), (testEndIdx+1):n]';
    testIdxs = [testStartIdx: testEndIdx]';
    nTe = testEndIdx - testStartIdx + 1;
    nTr = n - nTe;
    Xtr = X(trainIdxs, :);
    Ytr = Y(trainIdxs, :);
    Xte = X(testIdxs, :);
    Yte = Y(testIdxs, :);

    fprintf('CV iter %d/%d, nTRain: %d, nTest: %d\n', ...
      cvIter, params.numTrialsCV, nTr, nTe);

    % 2. Obtain Kernels and the Cholesky Decomposition
    [~, allKs] = kernelFunc(Xtr, Xtr);
    allLs = zeros(size(allKs));
    for j = 1:M
      allLs(:,:,j) = stableCholesky(allKs(:,:,j));
    end

    % 3. Obtain validation errors for each value of Lambda
    Beta = zeros(nTr, M); % Initialisation for largest Lambda
    % Optimise for each lambda
    for candIter = 1:numLambdaCands

      lambda = lambdaCands(candIter);
      fprintf('CViter: %d/%d, lambda = %0.5e\n', cvIter, params.numTrialsCV, ...
        lambda);

      % Call the optimisation routine
      params.initBeta = Beta;
      [Beta] = addKernelRegOpt(allLs, Ytr, decomp, lambda, params);

      % Some book-keeping to analyse sparsity.
      if candIter == 1
        % this gets overwritten at each iteration but that is ok.
        normBetaVals(candIter, :) = sqrt( sum(Beta.^2) );
      end

      % Now obtain the predictions and record the validation error
      Alpha = zeros(nTr, M);
      for j = 1:M
        Alpha(:,j) = (allLs(:,:,j)') \ Beta(:,j);
      end
      Ypred = getPrediction(Xte, Xtr, Alpha, kernelFunc);
      errorAccum(candIter) = errorAccum(candIter) + norm(Ypred - Yte).^2/nTe;

    end

  end
  %% Cross validation ends here
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Determine the best lambda value
  [~, bestLambdaIdx] = min(errorAccum);
  bestLambda = lambdaCands(bestLambdaIdx);

  % Perform Optimisation over all data points now
  [~, allKs] = kernelFunc(X, X);
  allLs = zeros(size(allKs));
  for j = 1:M
    allLs(:,:,j) = stableCholesky(allKs(:,:,j));
  end
  params.initBeta = zeros(n, M);
  params.maxNumIters = 10*params.maxNumIters;
  [optBeta, optStats] = addKernelRegOpt(allLs, Y, decomp, bestLambda, params);

  % Obtain the function handle
  optAlpha = zeros(n, M);
  for j = 1:M
    optAlpha(:,j) = (allLs(:,:,j)') \ optBeta(:,j);
  end
  predFunc = @(arg) getPrediction(arg, X, optAlpha, kernelFunc);

  % Before returning
  optStats.normBetaVals = normBetaVals;
  optStats.cvResults = [lambdaCands errorAccum/params.numTrialsCV];

  % print some summary statistics
  numSparseTerms = sum(sum(abs(optAlpha)) == 0);
  fprintf('Chosen lambda: %.5ef (%.5ef, %.5ef), sparsity: %d/%d\n\n', ...
    bestLambda, lambdaCands(1), lambdaCands(end), numSparseTerms, M);

end

