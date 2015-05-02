function pred = kernelRidgeRegression(Xte, Xtr, Ytr, lambda, method)
% Xte is a num_test_pts x num_dims matrix at which we need to estimate the
% function. 
% Xtr, Ytr: are the data matrix and regressors.
% method.kernel from {'RBF', 'Polynomial'}
% RBF:
% method.h
% Polynomial:
% method.degree
% method.bias 
n = size(Xtr,1);
if isequal(method.kernel, 'RBF')
    Ktr = GaussKernel(method.h, Xtr, Xtr);
    Kte = GaussKernel(method.h, Xtr, Xte);
elseif isequal(method.kernel, 'Polynomial')
    Ktr = PolyKernel(method.degree, method.bias, Xtr, Xtr);
    Kte = PolyKernel(method.degree, method.bias, Xtr, Xte);
end
    pred = (Ytr'* ((Ktr + lambda*eye(n))\Kte) )';


end

