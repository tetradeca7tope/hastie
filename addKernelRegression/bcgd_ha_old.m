function [optBeta, optStats] = bcgd_ha_old(Ls, Y, lambda, params)
    [n, ~, m] = size(Ls);
    y = Y/sqrt(n);
    K_all = Ls/sqrt(n);
    lambda_1 = 0; lambda_2 = 0;
    lambda_3 = lambda/m;
    params.initAlpha = params.initBeta;
    [optBeta, optStats] = inner(y, K_all, lambda_1, lambda_2, lambda_3, params);
    
end

function [alphas, stats] = inner(y, K_all, lambda_1, lambda_2, lambda_3, params)
    [n, ~, m] = size(K_all);
    params = processOptParamsCommon(params, n, m);
    max_iters = params.maxNumIters;
    max_ls_iters = 100;
    f_tol = params.tolerance;
 
    alphas = params.initAlpha;
    K_alphas = zeros(n,m);
    obj_history = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3);
    time_history = 0;
    startTime = cputime;
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
            
            step = 1;
            if params.backtracking  
                f_g = 0.5*norm(y_minus_other-K_alpha_g,2)^2 + ...
                    alpha_g'*(lambda_1/2*K_alpha_g + lambda_2/2*alpha_g) + ...
                    lambda_3*norm(alpha_g,2);
                delta = 1*grad_g'*d_g + norm(alpha_g + d_g,2) - norm(alpha_g, 2);
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
            end
            alphas(:,g) = alpha_g + step*d_g;
            K_alphas(:,g) = K_g*alphas(:,g);
        end

        obj = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3);
        currTime = cputime - startTime;
        obj_history = [obj_history; obj];
        time_history = [time_history; currTime];

        if params.verbose & mod(iter, params.verbosePerIter) == 0
          fprintf('#%d (%.4f): currObj: %0.5f\n', ...
            iter, currTime, obj);
        end


        if abs((obj-obj_history(end-1))/obj_history(end-1)) <= f_tol
          fprintf('Terminating BCGD-HA after %d iterations.\n', iter);
            break;
        end
    end
    stats.objective = obj_history;
    stats.time = time_history;
end

function [obj] = fast_objective(alphas, y, K_alphas, lambda_1, lambda_2, lambda_3)
% Complexity O(nm)
    m = size(K_alphas, 2); 
    fit = y - sum(K_alphas, 2);
    obj_fit = 0.5*(fit'*fit);
    obj_1 = 0;
    for g=1:m
        alpha_g = alphas(:,g);
        obj_1 = obj_1 + 0.5*lambda_1*alpha_g'*K_alphas(:,g);
    end

    obj_2 = 0;
    for g=1:m
        norm_2 = alphas(:,g)'*alphas(:,g);
        obj_2 = obj_2 + 0.5*lambda_2*norm_2;
    end

    obj_3 = 0;
    for g=1:m
        obj_3 = obj_3 + lambda_3*norm(alphas(:,g),2);
    end 
    obj = obj_fit + obj_1 + obj_2 + obj_3;
    %fprintf('fit=%f 1=%f 2=%f 3=%f sum=%f \n', ...
    %    obj_fit, obj_1, obj_2, obj_3, obj);
end
