function predFunc = svmRegWrap(X, Y, method)
% A wrapper for libsvm's regression function

  % prelims
  [n, D] = size(X);

  % determine parameters for svm
  h = 1.5 * norm(std(X)) * n^(-4/(4+D));
  gamma = 1/(2*h^2);

  if strcmp(method, 'eps'), s = 3;
  elseif strcmp(method, 'nu'), s = 4;
  end
  % -s 4: nu-SVR, -s 3: epsilon-SVR
  % -t 2: radial basis function kernel
  % -v 5: 5 fold cross validation
  % -q: quiet

  numCCands = 20;
  cCands = D * logspace(-3, 2, numCCands)';
  cErrs = zeros(numCCands, 1);

  for i = 1:numCCands
    libsvmArgs = sprintf('-s %d -g %.6f -t 2 -c %.5f -p 1 -v 5 -q', ...
      s, gamma, cCands(i) );
    cErrs(i) = svmtrain(Y, X, libsvmArgs);
  end

  [~, bestCIdx] = min(cErrs);
  bestC = cCands(bestCIdx);
  fprintf('%s-SVR: bestC = %.4f, (%.4f, %4f)\n', ...
    method, bestC, cCands(1), cCands(numCCands));

  % Now train again
  libsvmArgs = sprintf('-s %d -g %.6f -t 2 -c %.5f -p 1 -q', ...
    s, gamma, bestC );
  model = svmtrain(Y, X, libsvmArgs);
  predFunc = @(arg) svmpredict( ones(size(arg,1),1), arg, model);

end

