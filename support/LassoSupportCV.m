function LassoSupportCV();
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
            load('support-n%i-rep%i.mat', 'X', 'Y');
            st = LassoCV(X, Y, foldCV);
            f1_curr = [f1_curr st.f1];
            tpr_curr = [tpr_curr st.tpr];
            fpr_curr = [fpr_curr st.fpr];
            precision_curr = [precision_curr st.precision];
        end
        stats.f1_mean_list(n_ix) = mean(f1_curr);
        stats.f1_std_list(n_ix) = std(f1_curr);
    end
    figure; errorbar(stats.n_list,stats.f1_mean_list,stats.f1_std_list);
    figure; errorbar(stats.n_list,stats.tpr_mean_list,stats.tpr_std_list);
    figure; errorbar(stats.n_list,stats.fpr_mean_list,stats.fpr_std_list);
    figure; errorbar(stats.n_list,stats.precision_mean_list,stats.precision_std_list);
    save('lasso-stats.mat', 'stats');
end


function [st] = LassoCV(X, Y, foldCV)
    [n, p] = size(X);
    true_1d_sup = (1:4)';
    true_2d_sup = (5:12)';
    true_sup = union(true_1d_sup, true_2d_sup);
    false_sup = setdiff((1:p)', true_sup);
    [B, FitInfo] = lasso(X,Y,'CV', foldCV);
    b = B(:,FitInfo.IndexMinMSE);
    recovered = find(b ~= 0);
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
