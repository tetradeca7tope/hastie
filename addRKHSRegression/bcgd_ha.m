function [alphas, stats] = bcgd_ha(y, K_all, lambda_1, lambda_2, lambda_3)
    [n, ~, m] = size(K_all);
    max_iters = 100;
    max_ls_iters = 100;
    f_tol = 1.0e-4;
 
    alphas = zeros(n, m);
    K_alphas = zeros(n,m);
    obj_history = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3)
    time_history = 0;
    tic;
    
    % Precompute constant Hessian
    hess_all = zeros(n, n, m);
    h_all = zeros(m,1);
    for g=1:m
        K_g = K_all(:,:,g);
        hess_g = K_g'*K_g + lambda_1*K_g + lambda_2*eye(n,n);
        hess_all(:,:,g) = hess_g;
        h_all(g) = max(max(diag(hess_g)), 0.1);
    end
    
    for iter=1:max_iters
        for g=randperm(m)
            alpha_g = alphas(:,g);
            K_alpha_g = K_alphas(:,g);
            K_g = K_all(:,:,g);
            y_minus_other = y - sum(K_alphas(:,setdiff(1:m,g)),2);
            K_g_y_minus = K_g'*y_minus_other;
            grad_g = -K_g_y_minus + K_g'*K_alpha_g + ...
                    lambda_1*K_alpha_g + lambda_2*alpha_g;
            if norm(K_g_y_minus,2) <= lambda_3
                d_g = -alpha_g;
            else
                grad_minus_h = -grad_g + h_all(g)*alpha_g;
                d_g = 1/h_all(g)*(-grad_g - lambda_3*grad_minus_h/norm(grad_minus_h,2));
            end
            
            f_g = 0.5*norm(y_minus_other-K_alpha_g,2)^2 + ...
                alpha_g'*(lambda_1/2*K_alpha_g + lambda_2/2*alpha_g) + ...
                lambda_3*norm(alpha_g,2);
            delta = 1*grad_g'*d_g + norm(alpha_g + d_g,2) - norm(alpha_g, 2);
            assert(delta < 0);
            step = 1;
            success = 0;
            for lsiter=1:max_ls_iters
                alpha_step = alpha_g + step*d_g;
                K_alpha_step = K_g*alpha_step;
                f_step = 0.5*norm(y_minus_other-K_alpha_step,2)^2 + ...
                    alpha_step'*(lambda_1/2*K_alpha_step + lambda_2/2*alpha_step) + ...
                    lambda_3*norm(alpha_step,2);
                if f_step <= f_g + 0.001*step*delta
                    success = 1;
                    break;
                end
                step = 0.5*step;
            end
            assert(success == 1);
            alphas(:,g) = alpha_g + step*d_g;
            K_alphas(:,g) = K_g*alphas(:,g);
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
