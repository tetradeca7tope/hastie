function [K_all] = compute_kernels()
    % TODO- make symmetric 
    sigma = 1;
    load('data.mat', 'y', 'X');
    [n, p] = size(X);
    m = p*(p-1)/2;
    K_all = zeros(n,n,m);
    
    left = 1;
    right = 2;
    for k=1:p*(p-1)/2
        X_k = X(:,[left right]);
        K_k = X_k*X_k'/sigma^2;
        d_k = diag(K_k);
        K_k = exp(K_k - ones(n,1)*d_k'/2 - d_k*ones(1,n)/2);
        K_k_t = triu(K_k,1);
        K_k = triu(K_k,0) + triu(K_k,1)';
        eigen = eig(K_k);
        assert(isreal(eigen));
        min_eigen = min(eigen)
        %figure;
        %imagesc(K_k);
        K_all(:,:,k) = K_k; 
        right = right+1;
        if right > p
            left = left+1;
            right = left + 1;
        end
    end
    save('kernels.mat', 'K_all');
end
