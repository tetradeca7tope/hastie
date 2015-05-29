function predFunc = backFitting(Xtr, Ytr)
% Fits a sum of one dimensional functions.

  [n, D] = size(Xtr);
  mu = mean(Ytr);
  maxIters = min(20, round(1000/D));

  allDims = 1:D;
  fhatTr = zeros(n, D);
  predFuncs = cell(D, 1);

  for iter = 1:maxIters
    for i = 1:D
      otherIdxs = setdiff(allDims, i);
      Yi = Ytr - mu - sum(fhatTr(:,otherIdxs), 2);

%       % Splines
%       pp = spline(Xtr(:,i), Yi); predFunci = @(arg) ppval(pp, arg);

      % NW
      silvBw = 1.06 * std(Xtr(:,i)) * n^0.2;
      predFunci = @(arg) localPolyRegression(arg, Xtr(:,i), Yi, silvBw, 0);

      % Obtain fhati
      fhati = predFunci(Xtr(:,i));
      % update the prediction functions
      predFuncs{i} = @(arg) predFunci(arg) - mean(fhati);
      fhatTr(:,i) = fhati - mean(fhati);
    end
  end

  % Now return the function
  predFunc = @(arg) obtainPredictions(arg, predFuncs, mu);

end


% Prediction at a test point
function YPred = obtainPredictions(Xte, predFuncs, mu)
  [n, D] = size(Xte);
  YPred = mu*ones(n,1);
  for i = 1:D
    YPred = YPred + predFuncs{i}(Xte(:,i));
  end
end
