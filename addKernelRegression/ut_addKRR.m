% Unit test for addKRR

clear all;
close all;
clc;
addpath ../utils/
addpath ../otherMethods/
warning off;
rng('default');

% numDims = 100; n = 800; 
% numDims = 20; n = 1000; 
numDims = 20; n = 300; 

% Set up
% [func, funcProps] = getAdditiveFunction(numDims, numDims);
[func, funcProps] = getAdditiveFunction(numDims, round(numDims/2) -1);
% [func, funcProps] = getAdditiveFunction(numDims, 10);
[func, funcProps] = getAdditiveFunction(numDims, 4);
bounds = funcProps.bounds;
nTest = 1000;

% Sample train and test data uniformly within the bounds
Xtr = bsxfun(@plus, ...
  bsxfun(@times, rand(n, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Xte = bsxfun(@plus, ...
  bsxfun(@times, rand(nTest, numDims), (bounds(:,2) - bounds(:,1))' ), ...
  bounds(:, 1)' );
Ytr = func(Xtr);
Yte = func(Xte);

addKRRparams = struct();
% addKRRparams.orderCands = [79]';

% Now do prediction
tic,
predFunc = addKRR(Xtr, Ytr, addKRRparams);
toc,
Ypred = predFunc(Xte);
addErr = norm(Ypred - Yte)^2/nTest,

% Kernel Ridge Regression
tic,
krrPred = kernelRidgeReg(Xtr, Ytr, struct());
Ykrr = krrPred(Xte);
krrErr = norm(Ykrr - Yte)^2/nTest,

% Nadaraya Watson Regression
nwPred = localPolyRegressionCV(Xtr, Ytr, [], 0);
YNW = nwPred(Xte);
nwErr = norm(YNW - Yte)^2/nTest,

% Locally Quadratic Regression
lqPred = localPolyRegressionCV(Xtr, Ytr, [], 1);
Ylq = lqPred(Xte);
lqErr = norm(Ylq - Yte)^2/nTest,

% Additive GP
YGP = gpRegWrap(Xtr, Ytr, Xte);
GPErr = norm(YGP - Yte)^2/nTest,

