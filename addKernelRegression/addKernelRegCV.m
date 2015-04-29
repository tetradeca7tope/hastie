function [predFunc, decomposition, optStats] = ...
  addKernelRegCV(X, Y, decomposition, lambdaRange, params)
% Performs Additive RKHS regression and selects the optimal hyper parameters via
% cross validation.
% X, Y: Covariates and Labels
% decomposition is a struct containing the groups
% lambdaRange: range of values for regularization penalty

  % prelims
  n = size(X, 1);
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
    params.numLambdaCands = 100;
  end
  % Copy over to workspace
  numPartsKFoldCV = params.numPartsKFoldCV;
  numLambdaCands = params.numLambdaCands;

  % Set things up for cross validation
  errorAccum = zeros(numLambdaCands, 1);

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

    % 2. Obtain Kernels and the Cholesky Decomposition
    [kernelFunc, decomposition] = kernelSetup(X, decomposition);
    Ktr = 

    initBeta = zeros(n, 1);
    

  end

  

end



end


