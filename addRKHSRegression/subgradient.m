function [alphas,stats] = subgradient(y, K_all, lambda_1, lambda_2, lambda_3)
    [n, ~, m] = size(K_all);
    maxiters = 250;
    f_tol = 1.0e-4;
    gamma_0 = 1; 
    
    alphas = zeros(n, m);
    K_alphas = zeros(n, m);
    obj_history = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3);
    time_history = 0;
    tic;
    for iter=1:maxiters       
        
        subgrads = zeros(n,m);
        for g=randperm(m)
            alpha_g = alphas(:,g);
            K_g = K_all(:,:,g);
            K_alpha_g = K_alphas(:,g);
            y_minus_other = y - sum(K_alphas(:,setdiff(1:m,g)),2);
            subgrad = -K_g*y_minus_other + (K_g+lambda_1)*K_alpha_g+lambda_2*alpha_g;
            norm_g = norm(alpha_g,2);
            if norm_g ~= 0
                subgrad = subgrad + lambda_3/norm_g*alpha_g;
            end
            subgrads(:,g) = subgrad;
        end
        
        gamma_iter = gamma_0/iter;
        t_iter = gamma_iter/norm(subgrads(:),2);
        alphas = alphas - t_iter*subgrads;
        for g=1:m
            K_alphas(:,g) = K_all(:,:,g)*alphas(:,g);
        end
        
        obj = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3)
        obj_history = [obj_history obj];
        time_history = [time_history toc];
        if abs((obj-obj_history(end-1))/obj_history(end-1)) <= f_tol
            break;
        end
    end
    stats.objective = obj_history;
    stats.time = time_history;
end
