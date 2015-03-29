function val = l21Norm(X)
% Computes the L-21 Norm of the Matrix X
  val = sum( sqrt( sum(X.^2) ) );
end

