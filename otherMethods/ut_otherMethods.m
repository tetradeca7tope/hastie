% Unit test for other methods

clear all;
close all;
addpath ../utils/
addpath ~/libs/libsvm/matlab/  % add libsvm path here
addpath ~/libs/gpml/, startup; % add gpml path.
rng('default');

% method = 'NW';
% method = 'localLinear';
% method = 'localQuadratic';
% method = 'KRR';
method = 'GP';
method = 'addGP';
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
    Ypred = predFunc(Xte);

  case 'localLinear'
    predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 1);
    Ypred = predFunc(Xte);

  case 'localQuadratic'
    predFunc = localPolyKRegressionCV(Xtr, Ytr, [], 2);
    Ypred = predFunc(Xte);

  case 'KRR'
    predFunc = kernelRidgeReg(Xtr, Ytr, struct());
    Ypred = predFunc(Xte);

  case 'nuSVR'
    predFunc = svmRegWrap(Xtr, Ytr, 'nu');
    Ypred = predFunc(Xte);

  case 'epsSVR'
    predFunc = svmRegWrap(Xtr, Ytr, 'eps');
    Ypred = predFunc(Xte);

  case 'GP'
    Ypred = gpRegWrap(Xtr, Ytr, Xte);

  case 'addGP'
    Ypred = addGPRegWrap(Xtr, Ytr, Xte);

end

% Print results out
predErr = sqrt( norm(Ypred - Yte).^2 /nTest );
fprintf('Method : %s, predErr: %.4f, Ystd: %0.4f\n', method, predErr, std(Yte));

