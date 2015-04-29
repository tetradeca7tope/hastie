function obj = computeObjBeta(Beta, Ls, Y, lambda)
% Computes the objective in terms of Beta. beta_j = Lj * alpha
  M = size(Ls, 2);
  g = computeSmoothObjGradBeta(Beta, Ls, Y);
  h = (lambda/M) * l21Norm(Beta);
  obj = g + h;
end

