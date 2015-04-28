function obj = computeObj(Alpha, Ks, Y, lambda)
% Computes the objective in terms of Beta. beta_j = Lj * alpha
  M = size(Ks, 2);

  g = computeSmoothObjGrad(Alpha, Ks, Y, lambda1);
  h = (lambda/M) * l21Norm(Alpha);
  obj = g + h;
end

