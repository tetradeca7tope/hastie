function obj = computeObj(Alpha, Ks, Y, lambda1, lambda2)
% Computes the objective
  g = computeSmoothObjGrad(Alpha, Ks, Y, lambda1);
  h = lambda2 * l21Norm(Alpha);
%   g, h,
  obj = g + h;
end

