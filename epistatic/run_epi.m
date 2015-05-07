addpath ../utils/
addpath ../addKernelRegression/
load('epidata.mat', 'insulin_data', 'snp_data', 'meta');
decomposition.setting = 'maxGroupSize';
decomposition.maxGroupSize = 2;
params.optMethod = 'bcgdDiagHessian';
params.stepSizeCriterion = 'backTracking';
params.optVerbosePerIter = 5;
params.maxNumIters = 20;
params.optVerbose = true;
params.tolerance = 1.0e-5;
lambda = 15;
X = snp_data;
Y = insulin_data - mean(insulin_data);
[predFunc, optAlpha, optBeta, optStats, decomposition] = ...
  addKernelRegTrainOnly(X, Y, decomposition, lambda, params);
beta_norms = arrayfun(@(col) norm(optBeta(:,col)), 1:size(optBeta,2));
nonzero_groups = decomposition.groups(beta_norms~=0);
for i=1:size(nonzero_groups,1)
    nnz_g = nonzero_groups{i};
    if size(nnz_g,2)==1 % individual SNP
        s1 = nnz_g(1);
        fprintf('single SNP: chrom%i pos:%.2g \n', ...
            meta.snp_chromosomes(s1), meta.snp_positions(s1));
    else
        s1 = nnz_g(1);
        s2 = nnz_g(2);
        fprintf('pair SNP: chrom%i pos:%i and chrom%i pos:%.2g \n', ...
            meta.snp_chromosomes(s1), meta.snp_positions(s1), ...
            meta.snp_chromosomes(s2), meta.snp_positions(s2));

    end
end