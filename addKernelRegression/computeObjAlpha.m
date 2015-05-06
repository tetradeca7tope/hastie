function obj = computeObj(Alpha, Ks, Y, lambda)
% Computes the objective
  M = size(Ks, 3);
  [g, KAlpha] = computeSmoothObjGradAlpha(Alpha, Ks, Y);
  h = 0;
  for j = 1:M
    h = h + sqrt(Alpha(:,j)'*KAlpha(:,j));
  end
  h = (lambda/M) * h;
  obj = g + h;
end

