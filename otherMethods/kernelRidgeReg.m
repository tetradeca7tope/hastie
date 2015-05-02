function predFunc = kernelRidgeReg(Xtr, Ytr, params)
% Performs Kernel Ridge Regression. Cross validates for penalty parameter
% lambda.

  % prelims
  [n, D] = size(Xtr);

  % Cross validation parameters

  % First obtain the kernels
  decomposition.setting = 'groups';
  decomposition.groups = {[1:D]};
  kernelFunc = kernelSetup(Xtr, Ytr, decomposition);

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
    params.lambdaRange = [1e-3 10] * norm(std(X));
  end



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

    % Obtain the coefficients


  end


end


% Predictions for Kernel Ridge Regression
function preds = predictKRR(Xte, Xtr, kernelFunc, Ytr, alpha)
  Ktetr = kernelFunc(Xte, Xtr);
  preds = Ktetr * alpha;
end

