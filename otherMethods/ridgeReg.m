function predFunc = ridgeReg(X, Y)

  [n, D] = size(X);
  numLambdaCands = 25;
  lambdaCands = logspace(-4, 1, numLambdaCands) * norm(std(X))/sqrt(D);
  validErrs = zeros(numLambdaCands, 1);

  for candIter = 1:numLambdaCands
    validErrs(candIter) = crossValidate(X, Y, lambdaCands(candIter));
  end

  [~, bestLambdaIdx] = min(validErrs);
  bestLambda = lambdaCands(bestLambdaIdx);
  fprintf('Lin Reg: chose lambda = %.4f, (%.4f, %.4f)\n', bestLambda, ...
    lambdaCands(1), lambdaCands(end) );

  % Now obtain w
  w = (X'*X + bestLambda * eye(D)) \ X' * Y;
  predFunc = @(arg) arg * w;

end


function validErr = crossValidate(X, Y, lambda)

  numPartsKFoldCV = 5;
  numTrialsCV = 2;

  validErr = 0;
  [n, D] = size(X);

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

    % obtain coefficients
    w = (Xtr'*Xtr + lambda * eye(D)) \ Xtr' * Ytr;
    YPred = Xte*w;
    validErr = validErr + norm(YPred - Yte).^2/nTe; 
  end

end
