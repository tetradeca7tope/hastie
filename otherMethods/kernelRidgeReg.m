function predFunc = kernelRidgeReg(X, Y, params)
% Performs Kernel Ridge Regression. Cross validates for penalty parameter
% lambda.

  % prelims
  [n, D] = size(X);

  % Shuffle the data
  shuffleOrder = randperm(n);
  X = X(shuffleOrder, :);
  Y = Y(shuffleOrder, :);

  % First obtain the kernels
  decomposition.setting = 'groups';
  decomposition.groups = {[1:D]};
  kernelFunc = kernelSetup(X, Y, decomposition);
  if isfield(params, 'kernel') && isequal(params.kernel,'Polynomial') ...
    if ~isfield(params, 'bias')
      params.bias = 1;
    end
    if ~isfield(params, 'degree')
      params.degree = 2;
    end
    kernelFunc = kernelSetupPoly(X, Y, decomposition, params);
  end

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
  if ~isfield(params, 'lambdaRange')
    params.lambdaRange = [1e-4 100] * n;
  end

  % Determine candidates for lambda
  lambdaCands = logspace( log10(params.lambdaRange(1)), ...
    log10(params.lambdaRange(2)), params.numLambdaCands );
  validErrs = zeros(params.numLambdaCands, 1);

  for candIter = 1:params.numLambdaCands
    validErrs(candIter) = crossValidate(X, Y, kernelFunc, ...
      lambdaCands(candIter), params.numPartsKFoldCV, params.numTrialsCV);
  end

  % choose the best lambda
  [~, bestLambdaIdx] = min(validErrs);
  bestLambda = lambdaCands(bestLambdaIdx);
  fprintf('KRR: chose lambda = %.4f, (%.4f, %.4f)\n', bestLambda, ...
    lambdaCands(1), lambdaCands(end));

  % Now obtain the alphas
  K = kernelFunc(X, X);
  alpha = (K + bestLambda * eye(n)) \ Y;
  predFunc = @(arg) predictKRR(arg, X, kernelFunc, Y, alpha); 

end


% Do cross validation here
function validErr = crossValidate(X, Y, kernelFunc, lambda, ...
  numPartsKFoldCV, numTrialsCV)

  validErr = 0;
  n = size(X, 1);
  for cvIter = 1:numTrialsCV
    testStartIdx = round( (cvIter-1)*n/numPartsKFoldCV + 1);
    testEndIdx = round( cvIter*n/numPartsKFoldCV );
    trainIdxs = [1:(testStartIdx-1), (testEndIdx+1):n]';
    testIdxs = [testStartIdx:testEndIdx]';
    nTe = testEndIdx - testStartIdx + 1;
    nTr = n - nTe;
    Xtr = X(trainIdxs, :);
    Xte = X(testIdxs, :);
    Ytr = Y(trainIdxs, :);
    Yte = Y(testIdxs, :);

    % Obtain the coefficients
    K = kernelFunc(Xtr, Xtr);
    alpha = (K + lambda * eye(nTr))\Ytr;
    preds = predictKRR(Xte, Xtr, kernelFunc, Ytr, alpha);
    validErr = validErr + norm(preds - Yte).^2/nTe;
  end
end


% Predictions for Kernel Ridge Regression
function preds = predictKRR(Xte, Xtr, kernelFunc, Ytr, alpha)
  Ktetr = kernelFunc(Xte, Xtr);
  preds = Ktetr * alpha;
end

