function AddKernelSupportCV();
    addpath(genpath('../'));
    foldCV = 5;
    stats.n_list = [100 200 300 400 500 600 700 800 900 1000]';
    num_n = length(stats.n_list);
    num_reps = 20;
    stats.f1_mean_list = zeros(num_n,1);
    stats.f1_std_list = zeros(num_n,1);
    stats.tpr_mean_list = zeros(num_n,1);
    stats.tpr_std_list = zeros(num_n,1);
    stats.fpr_mean_list = zeros(num_n,1);
    stats.fpr_std_list = zeros(num_n,1);
    stats.precision_mean_list = zeros(num_n,1);
    stats.precision_std_list = zeros(num_n,1);
    for n_ix=1:num_n
        n = stats.n_list(n_ix);
        f1_curr = [];
        tpr_curr = [];
        fpr_curr = [];
        precision_curr = [];
        for rep=1:num_reps
            fname = sprintf('support-n%i-rep%i.mat', n, rep);
            fprintf('n:%i rep:%i \n', n, rep);
            load(fname, 'X', 'Y');
            st = AddKernelCV(X, Y, foldCV);
            f1_curr = [f1_curr st.f1];
            tpr_curr = [tpr_curr st.tpr];
            fpr_curr = [fpr_curr st.fpr];
            precision_curr = [precision_curr st.precision];
        end
        stats.f1_mean_list(n_ix) = mean(f1_curr);
        stats.f1_std_list(n_ix) = std(f1_curr);
        stats.tpr_mean_list(n_ix) = mean(tpr_curr);
        stats.tpr_std_list(n_ix) = std(tpr_curr);
        stats.fpr_mean_list(n_ix) = mean(fpr_curr);
        stats.fpr_std_list(n_ix) = std(fpr_curr);
        stats.precision_mean_list(n_ix) = mean(precision_curr);
        stats.precision_std_list(n_ix) = std(precision_curr);

    end
    figure; errorbar(stats.n_list,stats.f1_mean_list,stats.f1_std_list);
    figure; errorbar(stats.n_list,stats.tpr_mean_list,stats.tpr_std_list);
    figure; errorbar(stats.n_list,stats.fpr_mean_list,stats.fpr_std_list);
    figure; errorbar(stats.n_list,stats.precision_mean_list,stats.precision_std_list);
    save('AddKernel-stats.mat', 'stats');
end

function [st, optLambda] = AddKernelCV(X, Y, foldCV)
    [n, p] = size(X);
    true_1d_sup = (1:4)';
    true_2d_sup = (5:12)';
    true_sup = union(true_1d_sup, true_2d_sup);
    false_sup = setdiff((1:p)', true_sup);
    lambdaCands = sort(logspace(-3,3,10),'descend');
    numLambdaCands = length(lambdaCands);
    errorAccum = zeros(numLambdaCands,1);

    for cvIter = 1:foldCV
        testIdxs = randsample(1:n,floor(n/5));
        trainIdxs = setdiff(1:n, testIdxs);
        nTr = length(trainIdxs);
        nTe = length(testIdxs);
        
        Xtr = X(trainIdxs, :);
        Ytr = Y(trainIdxs, :);
        Xte = X(testIdxs, :);
        Yte = Y(testIdxs, :);

        fprintf('CV iter %d/%d\n', ...
            cvIter, foldCV);

        % 2. Obtain Kernels and the Cholesky Decomposition
        dc1.setting = 'groupSize';    dc1.groupSize    = 2; 
        [kernelFunc, dc1] = kernelSetup(X, Y, dc1);
        [~, allKs] = kernelFunc(Xtr, Xtr);
        allLs = zeros(size(allKs));
        M = size(allKs,3);
        for j = 1:M
            allLs(:,:,j) = stableCholesky(allKs(:,:,j));
        end

        % 3. Obtain validation errors for each value of Lambda
        Beta = zeros(nTr, M); % Initialisation for largest Lambda
        % Optimise for each lambda
        for candIter = 1:numLambdaCands
            lambda = lambdaCands(candIter);

            % Call the optimisation routine
            params.initBeta = Beta;
            params.setSizeCriterion = 'noback';
            params.tolerance = 1e-3;
            params.maxNumIters = 50;
            params.optMethod = 'bcgdha';
            [Beta, ~] = bcgd_ha(allLs, Ytr, lambda, params); 
            Alpha = zeros(nTr, M);
            for j = 1:M
               Alpha(:,j) = (allLs(:,:,j)') \ Beta(:,j);
            end
            Ypred = getPrediction(Xte, Xtr, Alpha, kernelFunc);
            errorAccum(candIter) = errorAccum(candIter) + norm(Ypred - Yte).^2/nTe;
        end
    end
      % Determine the best lambda value
      [~, bestLambdaIdx] = min(errorAccum);
      bestLambda = lambdaCands(bestLambdaIdx);

      % Perform Optimisation over all data points now
      [~, allKs] = kernelFunc(X, X);
      allLs = zeros(size(allKs));
      for j = 1:M
        allLs(:,:,j) = stableCholesky(allKs(:,:,j));
      end
      params.initBeta = zeros(n, M);
      params.maxNumIters = 10*params.maxNumIters;
      Beta = bcgd_ha(allLs, Y, lambda, params);
    recovered = find(Beta ~= 0);
    st.num_recovered = length(recovered);
    st.true_pos = length(intersect(recovered,true_sup));
    st.true_1d_pos = length(intersect(recovered,true_1d_sup));
    st.true_2d_pos = length(intersect(recovered,true_2d_sup));
    st.false_pos = length(setdiff(recovered, false_sup));
    st.tpr = st.true_pos / length(true_sup);
    st.tpr_1d = st.true_1d_pos / length(true_1d_sup);
    st.fpr = st.false_pos / length(false_sup);
    st.precision = length(st.true_pos) / length(recovered);
    st.f1 = 2*st.precision*st.tpr / (st.precision+st.tpr);
end
