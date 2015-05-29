function YPred = rbfInterpol(X, Y, Xte)
  model = rbfbuild(X,Y, 'G');
  YPred = rbfpredict(model, X, Xte);
%               'BH' = Biharmonic
%               'MQ' = Multiquadric
%               'IMQ' = Inverse Multiquadric
%               'TPS' = Thin plate spline
%               'G' = Gaussian
end

