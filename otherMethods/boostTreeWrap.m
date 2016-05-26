function YPred = boostTreeWrap(X, Y, Xte)
 
  % Use default values given in the test 
  opts.loss = 'squaredloss'; % can be logloss or exploss
  opts.shrinkageFactor = 0.01;
  opts.subsamplingFactor = 0.2;
  opts.maxTreeDepth = uint32(2);  % this was the default before customization
  opts.randSeed = uint32(rand()*1000);
  opts.verboseOutput = uint32(false);

  numIters = 600;
  model = SQBMatrixTrain(single(X), Y, uint32(numIters), opts);
  YPred = SQBMatrixPredict(model, single(Xte));

end

