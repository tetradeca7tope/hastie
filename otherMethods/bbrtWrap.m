function YPred = bbrtWrap(X, Y, Xte)

  leafNum = 4;
  numTrees = 100;
  brtModel = brtTrain( X, Y, leafNum, numTrees, 0.1);

  numTest = size(Xte, 1);
  YPred = zeros(numTest, 1);
  for i = 1:numTest
    YPred(i) = brtTest( Xte(i,:), brtModel, round(numTrees/2));
  end

end

