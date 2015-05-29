function predFunc = lassoWrap(X, Y)
  [B, FitInfo] = lasso(X, Y, 'CV', 5);
  [~, bestIdx] = min(FitInfo.MSE);
  beta = B(:, bestIdx);
  c = FitInfo.Intercept(bestIdx);
  predFunc = @(arg) arg*beta + FitInfo.Intercept(bestIdx);
end
