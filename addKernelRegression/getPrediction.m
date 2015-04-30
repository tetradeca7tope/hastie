function Ypred = getPrediction(Xte, X, Alpha, kernelFunc)
% Function to obtain the predictions using a given model (i.e. Alpha and
% Kernels)
  M = size(Alpha, 2);
  numTest = size(Xte, 1);
  Ypred = zeros(numTest, 1);

  [~, allKs] = kernelFunc(Xte, X);
  for j = 1:M
    Ypred = Ypred + allKs(:,:,j) * Alpha(:,j);
  end

end

