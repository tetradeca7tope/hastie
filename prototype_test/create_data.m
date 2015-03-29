function create_data(n, p, p_real)
    close all;
    X = randn(n, p);
    assert(p_real <= p);
    lefts = [(1:p) 1]';
    rights = [(2:p) p]';
    lefts = lefts(1:p_real);
    rights = rights(1:p_real);
    true_inter = X(:,lefts) .* X(:,rights);
    weight = 1/sqrt(p_real);
    w = weight*ones(p_real,1); 
    w(randsample(p_real,floor(p_real/2),false)) = -1*weight;
    sigma = 1;
    y = true_inter*w + sigma*randn(n,1);
    y = y - mean(y);
    
    all_inter = zeros(n,p*(p-1)/2);
    left = 1;
    right = 2;
    is_true = zeros(p*(p-1)/2,1);
    for i=1:p*(p-1)/2
        is_true(i) = any(ismember([lefts rights],[left right],'rows'));
            
        all_inter(:,i) = X(:,left).*X(:,right);
        right = right+1;
        if right > p
            left = left+1;
            right = left + 1;
        end
    end
    cov_all = cov([y all_inter]);
    cov_all = [max(cov_all(:))*[1 is_true']; cov_all];
    corr_all = corrcoef([y all_inter]);
    corr_all = [max(corr_all(:))*[1 is_true']; corr_all];

    figure;
    imagesc(cov_all);
    colorbar;
    figure;
    imagesc(corr_all);
    colorbar;
    save('data.mat', 'y', 'X');
end