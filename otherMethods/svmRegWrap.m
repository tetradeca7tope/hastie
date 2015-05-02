function predFunc = svmRegWrap(X, Y, params)
% A wrapper for libsvm's regression function

  libsvmArgs = '-v 5 -q'; % 5 fold CV and quiet
  model = svmtrain(Y, X, libsvmArgs);
  predFunc = @(arg) svmpredict( ones(size(arg,1),1), arg );

end

