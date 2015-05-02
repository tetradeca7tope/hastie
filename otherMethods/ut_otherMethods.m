% Unit test for other methods

clear all;
close all;
addpath ../utils/
addpath ~/libs/libsvm/matlab/  % add libsvm path here
rng('default');

% method = 'NW';
% method = 'localLinear';
% method = 'localQuadratic';
% method = 'KRR';
method = 'KRR-Poly';
% method = 'epsSVR';

% Set up
% numDims = 10; n = 100;
numDims = 20; n = 200;
[func, funcProps] = getAdditiveFunction(numDims, numDims);
bounds = funcProps.bounds;
nTest = n;

% Sample train and test data uniformly within the bounds
Xtr = bsxfun(@plus, ...
  bsxfun(@times, rand(n, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Ytr = func(Xtr);
Yte = func(Xte);


switch method

  case 'NW'
    predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 0);

  case 'localLinear'
    predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 1);

  case 'localQuadratic'
    predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 2);

  case 'KRR'
    predFunc = kernelRidgeReg(Xtr, Ytr, struct());
  case 'KRR-Poly'
    params.kernel = 'Polynomial';
    params.bias = 1;
    params.degree = 2;
    predFunc = kernelRidgeReg(Xtr, Ytr, params);
  case 'nuSVR'
    predFunc = svmRegWrap(Xtr, Ytr, 'nu');

  case 'epsSVR'
    predFunc = svmRegWrap(Xtr, Ytr, 'eps');

end

% Print results out
Ypred = predFunc(Xte);
predErr = sqrt( norm(Ypred - Yte).^2 /nTest );
fprintf('Method : %s, predErr: %.4f, Ystd: %0.4f\n', method, predErr, std(Yte));
