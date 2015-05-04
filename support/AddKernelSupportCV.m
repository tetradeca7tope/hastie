function AddKernelSupportCV();
    addpath(genpath('../'));
    foldCV = 5;
    stats.n_list = [100 200 300 400 500 600 700 800 900 1000]';
    stats.n_list = [100 200 400 600 800 1000]'; %TODO
    num_n = length(stats.n_list);
    num_reps = 3; %TODO
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
        save('AddKernel-stats.mat', 'stats');
    end
    figure('name', 'F1'); errorbar(stats.n_list,stats.f1_mean_list,stats.f1_std_list);
    figure('name', 'TPR'); errorbar(stats.n_list,stats.tpr_mean_list,stats.tpr_std_list);
    figure('name', 'FPR'); errorbar(stats.n_list,stats.fpr_mean_list,stats.fpr_std_list);
    figure('name', 'Precision'); errorbar(stats.n_list,stats.precision_mean_list,stats.precision_std_list);
    save('AddKernel-stats.mat', 'stats');
end

function [st, optLambda] = AddKernelCV(X, Y, foldCV)
    [n, p] = size(X);
    lambdaCands = sort(logspace(0,2,10),'descend');
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
        dc1.setting = 'maxGroupSize';    dc1.maxGroupSize = 2; 
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
            params.stepSizeCriterion = 'backTracking';
            params.tolerance = 1e-3;
            params.maxNumIters = 10;
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
    optLambda = lambdaCands(bestLambdaIdx)

    % True support
    true_groups = num2cell([5 6; 7 8; 9 10; 11 12],2);
    for i=1:4
        true_groups{i+4} = i;
    end
    true_sup_logical = zeros(dc1.M,1);
    for i=1:dc1.M
        for j=1:8
            if isequal(dc1.groups{i},true_groups{j})
                true_sup_logical(i) = 1;
            end
        end
    end
    true_sup = find(true_sup_logical);
    false_sup = setdiff((1:dc1.M)', true_sup);

    % Perform Optimisation over all data points now
    [~, allKs] = kernelFunc(X, X);
    allLs = zeros(size(allKs));
    for j = 1:M
        allLs(:,:,j) = stableCholesky(allKs(:,:,j));
    end
    params.initBeta = zeros(n, M);
    params.maxNumIters = 2*params.maxNumIters;
    Beta = bcgd_ha(allLs, Y, lambda, params);
    recovered_sup_logical = any(Beta,1)';
    recovered = find(recovered_sup_logical);
    st.num_recovered = length(recovered)
    st.true_pos = length(intersect(recovered,true_sup));
    st.false_pos = length(setdiff(recovered, false_sup));
    st.tpr = st.true_pos / length(true_sup);
    st.fpr = st.false_pos / length(false_sup);
    st.precision = length(st.true_pos) / length(recovered);
    st.f1 = 2*st.precision*st.tpr / (st.precision+st.tpr);
end
