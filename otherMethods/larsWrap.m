function predFunc = larsWrap(X, Y)
%   beta = lars(X, Y, 'lar', inf, true);
  beta = lars(X, Y, 'lar', inf, true);
%   beta = beta(end,:)';
  beta = beta(1,:)';
  meanY = mean(Y);
  predFunc = @(arg) arg*beta + meanY;
end

