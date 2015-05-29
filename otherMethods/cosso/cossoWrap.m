function Ypred = cossoWrap(Xtr, Ytr, Xte)
% My wrapper for Cosso
  nTe = size(Xte, 1);
  predtestandtheta = predadd(Xtr, Ytr, Xte, 5); 
  Ypred = predtestandtheta(1:nTe, 1);
end

