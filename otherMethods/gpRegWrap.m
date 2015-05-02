function [preds] = gpRegWrap(X, Y, Xte)
% A wrapper for GP Regression using GPML.

  likFunc = @likGauss;
  covFunc = @covSEiso;
  hyp2.cov = [0;0];
  hyp2.lik = log(0.1);

  hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covFunc, likFunc, X, Y);
  fprintf('Chosen GP hyperparameters: %.4f, %.4f\n', hyp2.cov(1), hyp2.cov(2));

  % Now do prediction
  preds = gp(hyp2, @infExact, [], covFunc, likFunc, X, Y, Xte); 

end
