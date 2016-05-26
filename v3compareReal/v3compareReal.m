% This script is for comparing different regression algorithms.
% We will try additive Kernel Ridge Regression, plain Ridge Regression and NW.

close all;
clear all;
% clc;
addpath ../addKernelRegression/
addpath ../utils/
addpath ../otherMethods/
addpath ../otherMethods/cosso/
addpath ../otherMethods/ARESLab/
addpath ../otherMethods/RBF/
addpath ../otherMethods/M5PrimeLab/
addpath ../otherMethods/sqb-0.1/build/
addpath ../otherMethods/BoostedRegressionTree/
addpath ~/libs/libsvm/matlab/
% addpath ~/libs/gpml/, startup;
rng('default');
warning off;

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
% dataset = 'blog';       % 
% dataset = 'LRGs';       % 
% dataset = 'Skillcraft';
% dataset = 'Skillcraft-small';
% dataset = 'Airfoil*';
% dataset = 'CCPP*';
% dataset = 'Insulin';
% dataset = 'CCPP*small';
% dataset = 'diabetes'; %TODO
% dataset = 'Bleeding';
% dataset = 'School';
% dataset = 'Brain';
% dataset = 'fMRI';

% Load data
[Xtr, Ytr, Xte, Yte] = getDataset(dataset);
[nTr, numDims] = size(Xtr);
nTe = size(Xte, 1);

% Save file name
saveFileName = sprintf('results/%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS') );

regAlgos = {}; cnt = 0;
% % Now add each method one by one
cnt=cnt+1; regAlgos{cnt}= {'addKRR',   @(X,Y,Xte) addKRR(X, Y)};
cnt=cnt+1; regAlgos{cnt}= {'KRR', @(X,Y,Xte) kernelRidgeReg(X, Y, struct())};
cnt=cnt+1; regAlgos{cnt}= {'KNN', @(X,Y,Xte) KnnRegressionCV(X, Y, [])};
cnt=cnt+1; regAlgos{cnt}= {'NW', @(X,Y,Xte) localPolyRegressionCV(X,Y,[],0)};
cnt=cnt+1; regAlgos{cnt}= {'LL', @(X,Y,Xte) localPolyRegressionCV(X,Y,[],1)};
cnt=cnt+1; regAlgos{cnt}= {'LQ', @(X,Y,Xte) localPolyRegressionCV(X,Y,[],2)};
% cnt=cnt+1; regAlgos{cnt}= {'LC', @(X,Y,Xte) localPolyRegressionCV(X,Y,[],3)};
cnt=cnt+1; regAlgos{cnt}= {'SV-eps', @(X,Y,Xte) svmRegWrap(X, Y, 'eps')};
cnt=cnt+1; regAlgos{cnt}= {'SV-nu', @(X,Y,Xte) svmRegWrap(X, Y, 'nu')};
cnt=cnt+1; regAlgos{cnt}= {'GP', @(X,Y,Xte) gpRegWrap(X,Y,Xte)};
cnt=cnt+1; regAlgos{cnt}= {'regTree', @(X,Y,Xte) regTree(X, Y, Xte)};
cnt=cnt+1; regAlgos{cnt}= {'GBRT', @(X,Y,Xte) bbrtWrap(X, Y, Xte)};
cnt=cnt+1; regAlgos{cnt}= {'RBFI',  @(X,Y,Xte) rbfInterpol(X,Y,Xte)};
cnt=cnt+1; regAlgos{cnt}= {'M5P',  @(X,Y,Xte) m5prime(X,Y,Xte)};
cnt=cnt+1; regAlgos{cnt}= {'Shepard', @(X,Y,Xte) shepard(X,Y,2,Xte)};
cnt=cnt+1; regAlgos{cnt}= {'BF', @(X,Y,Xte) backFitting(X,Y)};
cnt=cnt+1; regAlgos{cnt}= {'MARS', @(X,Y,Xte) mars(X,Y,Xte)};
% if numDims <= 40
%   cnt=cnt+1; regAlgos{cnt}= {'MARS', @(X,Y,Xte) mars(X,Y,Xte)};
% end 
cnt=cnt+1; regAlgos{cnt}= {'COSSO', @(X,Y,Xte) cossoWrap(X, Y, Xte)};
cnt=cnt+1; regAlgos{cnt}= {'spam', @(X,Y,Xte) SpamRegressionCV(X, Y)};
if numDims <= 15 & nTr < 300
  cnt=cnt+1; regAlgos{cnt}= {'addGP', @(X,Y,Xte) addGPRegWrap(X,Y,Xte)};
end
cnt=cnt+1; regAlgos{cnt}= {'LR',       @(X,Y,Xte) ridgeReg(X, Y)};
cnt=cnt+1; regAlgos{cnt}= {'LASSO',       @(X,Y,Xte) lassoWrap(X, Y)};
cnt=cnt+1; regAlgos{cnt}= {'LARS',       @(X,Y,Xte) larsWrap(X, Y)};
% % %   cnt=cnt+1; regAlgos{cnt}= {'addGP', @(X,Y,Xte) addGPRegWrap(X,Y,Xte)};
% % %   cnt=cnt+1; regAlgos{cnt}= {'MARS', @(X,Y,Xte) mars(X,Y,Xte)};


numRegAlgos = numel(regAlgos),

results = zeros(numRegAlgos, 1);
times = zeros(numRegAlgos, 1);

fprintf('Dataset: %s (n, D) = (%d, %d)\n=====================================\n', ...
dataset, nTr, numDims);
% Now run each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:numRegAlgos

  % Learn the algorithm
  startTime = cputime;
  if strcmp(regAlgos{i}{1}, 'addKRR')
    [predFunc, addKrrOrder] = regAlgos{i}{2}(Xtr, Ytr, Xte);
  else
    predFunc = regAlgos{i}{2}(Xtr, Ytr, Xte);
  end

  if strcmp(class(predFunc), 'double')
    YPred = predFunc;
  else
    YPred = predFunc(Xte);
  end
  times(i) = cputime - startTime;
  predError = norm(YPred-Yte).^2/nTe;
  results(i) = predError;
  fprintf('Method: %s, err: %.5f\n\n', regAlgos{i}{1}, predError);
end

% Save results
saveFileName = sprintf('results/real-%s-%s.mat', dataset, ...
  datestr(now, 'mmdd-HHMMSS'));
save(saveFileName, 'regAlgos', 'results', 'numRegAlgos', 'dataset');

printV3Results;

