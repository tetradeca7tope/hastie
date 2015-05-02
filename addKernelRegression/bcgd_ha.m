function [optBeta, optStats] = bcgd_ha(Ls, Y, lambda, params)
    [n, ~, m] = size(Ls);
    y = Y/sqrt(n);
    L_all = Ls/sqrt(n);
    Lambda = lambda/m;
    objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);

    params = processOptParamsCommon(params, n, m);
    max_ls_iters = 100;
 
    Beta = params.initBeta;
    L_Beta = zeros(n,m);
    for g=1:m
        L_Beta(:,g) = L_all(:,:,g)*Beta(:,g);
    end
    pred = sum(L_Beta,2);
    
    prevObj = objective(Beta);
    objHistory = prevObj;
    timeHistory = 0;
    startTime = cputime;
    
    % Precompute constant Hessian
    hess_all = zeros(n, n, m);
    h_all = zeros(m,1);
    for g=1:m
        L_g = L_all(:,:,g);
        hess_g = L_g'*L_g;
        hess_all(:,:,g) = hess_g;
        h_all(g) = max(max(diag(hess_g)), 0.1);
    end
    
    for iter=1:params.maxNumIters
        for g=randperm(m)
            Beta_g = Beta(:,g);
            L_Beta_g = L_Beta(:,g);
            L_g = L_all(:,:,g);
            y_minus_other = y - (pred - L_Beta(:,g));
            L_g_y_minus = L_g'*y_minus_other;
            grad_g = -L_g_y_minus + L_g'*L_Beta_g;
            if norm(L_g_y_minus,2) <= Lambda
                Beta(:,g) = 0;
                pred = pred - L_Beta_g;
                L_Beta(:,g) = 0;
            else
                grad_minus_h = -grad_g + h_all(g)*Beta_g;
                d_g = 1/h_all(g)*(-grad_g - Lambda*grad_minus_h/norm(grad_minus_h,2));
                step = 1;
                if params.backtracking  
                    f_g = 0.5*norm(y_minus_other-L_Beta_g,2)^2 + ...
                        Lambda*norm(Beta_g,2);
                    delta = 1*grad_g'*d_g + norm(Beta_g + d_g,2) - norm(Beta_g, 2);
                    success = 0;
                    for lsiter=1:max_ls_iters
                        Beta_step = Beta_g + step*d_g;
                        L_Beta_step = L_g*Beta_step;
                        f_step = 0.5*norm(y_minus_other-L_Beta_step,2)^2 + ...
                            Lambda*norm(Beta_step,2);
                        if f_step <= f_g + 0.001*step*delta
                            success = 1;
                            break;
                        end
                        step = 0.5*step;
                    end
                    assert(success == 1);
                end
                Beta(:,g) = Beta_g + step*d_g;
                L_Beta(:,g) = L_g*Beta(:,g);
                pred = pred + step*L_g*d_g;
            end
        end

        currObj = objective(Beta);
        currTime = cputime - startTime;
        objHistory = [objHistory; currObj];
        timeHistory = [timeHistory; currTime];

        if params.optVerbose && mod(iter, params.optVerbosePerIter) == 0
            fprintf('bcgdDH #%d (%.4f): currObj: %0.4e\n', ...
                iter, currTime, currObj);
        end

        if abs((currObj-prevObj)/prevObj) <= params.tolerance
            fprintf('Terminating BCGD-HA after %d iterations.\n', iter);
            break;
        end
        prevObj = currObj;
    end
    optStats.objective = objHistory;
    optStats.time = timeHistory;
    optBeta = Beta;
end

function [obj] = fast_objective(Beta, y, Ls, Lambda)
% Complexity O(nm)
    m = size(Ls, 2); 
    fit = y - sum(Ls, 2);
    obj_fit = 0.5*(fit'*fit);
    obj_norm = 0;
    for g=1:m
        obj_norm = obj_norm + Lambda*norm(Beta(:,g),2);
    end 
    obj = obj_fit + obj_norm;
end
