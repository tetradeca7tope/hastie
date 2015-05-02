function [preds] = addGPRegWrap(X, Y, Xte, order)
% A wrapper for GP Regression using GPML.

  [n, D] = size(X);
  if ~exist('order', 'var')
    order = 5;
  end

  likFunc = @likGauss;
  covFunc = {'covADD', {1:order, 'covSEiso'}};
  hyp2.lik = log(0.1);
  hyp2.cov = [log(ones(1,2*D)), log(ones(1,order))];

  hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covFunc, likFunc, X, Y);

  % Now do prediction
  preds = gp(hyp2, @infExact, [], covFunc, likFunc, X, Y, Xte); 

end
