    addpath(genpath('../'));
    fname = sprintf('support-n%i-rep%i.mat', n, 1);
    load(fname, 'X', 'Y');
    dc1.setting = 'maxGroupSize';    dc1.maxGroupSize = 2; 
    params.optMethod = 'bcgdha';
    [kernelFunc, dc1] = kernelSetup(X, Y, dc1);
    [~, allKs] = kernelFunc(X, X);
    M = dc1.M;

    % True support
    true_groups = num2cell([5 6; 7 8; 9 10; 11 12],2);
    for i=1:4
        true_groups{i+4} = i;
    end
    true_sup_logical = zeros(M,1);
    for i=1:M
        for j=1:8
            if isequal(dc1.groups{i},true_groups{j})
                true_sup_logical(i) = 1;
            end
        end
    end
    true_sup = find(true_sup_logical);
    numTrue = size(true_sup,1);
    false_sup = setdiff((1:M)', true_sup);


    allLs = zeros(size(allKs));
    for j = 1:M
        allLs(:,:,j) = stableCholesky(allKs(:,:,j));
    end
    Beta = zeros(n, M);
    params.initBeta = zeros(n, M);

    
    lambdaMax = 0;
    for g=1:M
        lambdaMax = max(lambdaMax, norm((allLs(:,:,g)'*Y)/sqrt(n),2))*M;
    end
    %lambdaMax = 300
    %lambdaMin = 50
    %lambdaCands = sort(logspace(log10(lambdaMin),log10(lambdaMax),numPath)','descend');
    lambdaCands = [1000; 600; 300; 200; 100];
    %lambdaCands = [100; 90; 85; 80; 75; 70; 50];
    numPath = length(lambdaCands); 
    all_norms = zeros(numPath,M);
    for l_ix=1:numPath
        lambda = lambdaCands(l_ix)
        params.initBeta = zeros(n, M);
        params.stepSizeCriterion = 'backTracking';
        params.tolerance = 1e-4;
        params.maxNumIters = 500;
        Beta = bcgd_ha(allLs, Y, lambda, params);
        beta_norms = arrayfun(@(col) norm(Beta(:,col)), 1:size(Beta,2));
        all_norms(l_ix,:) = beta_norms;
        numNonZero = length(find(beta_norms ~= 0))
    end

    nonzero_norms = all_norms(:,find(any(all_norms,1)));
    nonzero_truesup = true_sup_logical(any(all_norms,1));
    figure;
    hold on;
    for i=1:size(nonzero_norms,2)
        plot(lambdaCands,nonzero_norms(:,i),'k');
    end
    for i=true_sup'
        plot(lambdaCands,all_norms(:,i),'r');
        max(all_norms(:,i))
        i
    end
ylabel('||\beta||_2','FontSize', 16);
xlabel('\lambda','FontSize', 16);
