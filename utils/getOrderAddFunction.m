function [func, funcProps] = getOrderAddFunction(numDims, order)

  numCombs = nchoosek(numDims, order);
  combs = combnk(1:numDims, order);

  % Use this as a sub function for the three coordinates
  if order == 1
    [subFunc, subProperties] = get2Modal1DFunction;
  else
    [subFunc, subProperties] = get3ModalFunction(order);
  end

  % pass the bounds and the objective function
  funcProps.bounds = repmat([-1 1], numDims, 1);
  func = @(X) objFunction(X, subFunc, numDims, order);

end


% This is the function that does the computation
function val = objFunction(X, subFunc, numDims, order)
  numCombs = nchoosek(numDims, order);
  combs = combnk(1:numDims, order);
  val = 0;
  for i = 1:numCombs
    val = val + subFunc( X(:, combs(i, :)) );
  end
  val = val/numCombs;
end

