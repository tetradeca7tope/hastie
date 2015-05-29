function YPred = mars(X, Y, Xte)
% A wrapper for MARS
  model = aresbuild(X, Y);
  YPred = arespredict(model, Xte);
end

