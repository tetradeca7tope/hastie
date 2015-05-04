% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
clc;
addpath ../addKernelRegression/
addpath ../utils/
addpath ../otherMethods/
addpath ~/libs/libsvm/matlab/
addpath ~/libs/gpml/, startup;
rng('default');

% Determine dataset
% dataset = 'debug';
% dataset = 'parkinson21';              % **
% dataset = 'parkinson21-small';
% dataset = 'housing';                  % **
% dataset = 'music';                    % **
% dataset = 'music-small';                % **
% dataset = 'telemonitoring-total';
% dataset = 'telemonitoring-total-small';  %**
% dataset = 'telemonitoring-motor-small';  %**
% dataset = 'forestfires';       % **
% dataset = 'propulsion';       % xx
dataset = 'blog';       % 

% Load data
[Xtr, Ytr, Xte, Yte] = getDataset(dataset);
[nTr, numDims] = size(Xtr);
nTe = size(Xte, 1);

% Set the following two
decompRand.setting = 'randomGroups';
decompRand.groupSize = min(10, ceil(numDims/4));
decompRand.numRandGroups = min(200, 10*numDims);
optParamsRand.maxNumIters = 50;
optParamsRand.optMethod = 'bcgdDiagHessian';

decompEsp.setting = 'espKernel';
optParamsEsp.maxNumIters = 400;
optParamsEsp.optMethod = 'bcgdDiagHessian';

% Save file name
saveFileName = sprintf('results/%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS') );

regressionAlgorithms = { ...
%   {'addKrrRand',   @(X,Y,Xte) addKernelRegCV(X,Y,decompRand, [], optParamsRand)},...
  {'addKrrEsp',   @(X,Y,Xte) addKernelRegCV(X,Y,decompEsp, [], optParamsEsp)}, ...
  {'KRR',      @(X,Y,Xte) kernelRidgeReg(X, Y, struct())}, ...
  {'KNN',      @(X,Y,Xte) KnnRegressionCV(X, Y, [])}, ...
  {'NW',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],0)}, ...
  {'LL',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],1)}, ...
  {'LQ',       @(X,Y,Xte) localPolyKRegressionCV(X,Y,[],2)}, ...
  {'GP',       @(X,Y,Xte) gpRegWrap(X,Y,Xte)}, ...
  {'SVR',      @(X,Y,Xte) svmRegWrap(X, Y, 'eps')}, ...
%   {'spam',     @(X,Y,Xte) SpamRegressionCV(X, Y)}, ...
%   {'addGP',    @(X,Y,Xte) addGPRegWrap(X,Y,Xte)}, ...
  };

numRegAlgos = numel(regressionAlgorithms);


results = zeros(numRegAlgos, 1);

fprintf('Dataset: %s\n============================================\n', dataset);
% Now run each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:numRegAlgos
  predFunc = regressionAlgorithms{i}{2}(Xtr, Ytr, Xte);
  if strcmp(class(predFunc), 'double')
    YPred = predFunc;
  else
    YPred = predFunc(Xte);
  end
  predError = norm(YPred-Yte).^2/nTe;
  results(i) = predError;
  fprintf('Method: %s, err: %.4f\n', regressionAlgorithms{i}{1}, predError);
end

% Save results
saveFileName = sprintf('results/real-%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS'));
save(saveFileName, 'regressionAlgorithms', 'results', 'numRegAlgos');

printV3Results;

