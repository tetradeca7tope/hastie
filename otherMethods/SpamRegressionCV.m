function [predFunc, opt_lambda] = SpamRegressionCV( ...
  Xtr, Ytr, lambda_cands)
% This function implements locally Polynomial Kernel regression and searches for
% the optimal hyper-params (bandwidth and poly order)
% if h_cands is empty, picks 10 values based on the std of X. If polyOrder_cands
% is empty uses poly order 2.
% Picks the hyper parametsr using Kfold CV. Outputs predictions and the optimal
% parameters. For meanings of other variables read localPolyKRegression.m

  % prelims
  num_train_pts = size(Xtr, 1);
  num_dims = size(Xtr, 2);
  num_kfoldcv_partitions = min(10, num_train_pts);;

  % specify default values for candidats and poly order if not specified
  if ~exist('lambda_cands', 'var') | isempty(lambda_cands)
    lambda_cands = logspace(-6, 3, 30)';
  end
  num_lambda_cands = size(lambda_cands, 1);

  % Shuffle the data
  shuffle_order = randperm(num_train_pts);
  Xtr = Xtr(shuffle_order, :);
  Ytr = Ytr(shuffle_order, :);

  % Now iterate through these combinations and obtain the optimal value
  best_cv_error = inf;
  for lambda_iter = 1:num_lambda_cands
      curr_cv_error = KFoldExperiment(Xtr, Ytr, ...
            num_kfoldcv_partitions, lambda_cands(lambda_iter));      
      if best_cv_error >= curr_cv_error
        best_cv_error = curr_cv_error;
        opt_lambda = lambda_cands(lambda_iter);
      end
  end

  % Finally use the optimal parameters and all the data to fit a function
  fprintf('opt_lambda:%e\n', opt_lambda);
  function Ypred = spamPredict(arg);
    [~, Ypred] = spamRegression(arg, Xtr, Ytr, opt_lambda);
  end
  predFunc = @spamPredict;

end

function kfold_error = KFoldExperiment(X, y, num_partitions, lambda)
% This function computes the cross validation error for the current lambda

  m = size(X, 1);
  kfold_error = 0;

  for kfold_iter = 1:num_partitions
    test_start_idx = round( (kfold_iter-1)*m/num_partitions + 1 );
    test_end_idx   = round( kfold_iter*m/num_partitions );
    train_indices = [1:test_start_idx-1, test_end_idx+1:m];
    test_indices = [test_start_idx : test_end_idx];
    Xtr = X(train_indices, :);
    ytr = y(train_indices);
    Xte = X(test_indices, :);
    yte = y(test_indices);

    % obtain the predictions
    [~, pred] = SpamRegression(Xte, Xtr, ytr, lambda);
    % accumulate the errors
    kfold_error = kfold_error + sum( (yte - pred).^2 );
  end

end

