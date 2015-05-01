function z = l2NormProxOp(x, t)
% Prox operator for the l2 norm penalty 

  normX2 = norm(x,2);
  if normX2 > t,
    z = x - t * x /normX2;
  else
    z = zeros(size(x));
  end

end

