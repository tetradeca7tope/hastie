function YPred = m5prime(X, Y, Xte)
% A wrapper for MARS
  model = m5pbuild(X, Y);
  YPred = m5ppredict(model, Xte);
end

