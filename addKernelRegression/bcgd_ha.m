function [optBeta, optStats] = bcdExact(Ls, Y, lambda, params)
    [n, ~, m] = size(Ls);
    % Create a function handle for the objective
    objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
    smoothObjGrad = @(arg) computeSmoothObjGradBeta(arg, Ls, Y);
    % Obtain optimisation params
    params = processOptParamsCommon(params, n, m);
    params.max_ls_iters = 0;
    Beta = params.initBeta;

    % Set up book keeping
    prevObj = objective(Beta);
    objHistory = prevObj;
    timeHistory = 0;
    startTime = cputime;

    % Maintain L_g*Beta_g for all groups
    L_beta = zeros(n, m);
    for g=1:m
        L_beta(:,g) = Ls(:,:,g)*Beta(:,g);
    end

    % Precompute constant Hessian
    hess_all = zeros(n, n, m);
    h_all = zeros(m,1);
    for g=1:m
        L_g = Ls(:,:,g);
        hess_g = 1/(2*n)*L_g'*L_g;
        hess_all(:,:,g) = hess_g;
        h_all(g) = max(max(diag(hess_g)), 0.1);
    end
    
    for iter=1:params.maxNumIters
        for g=randperm(m)
            Beta_g = Beta(:,g);
            L_beta_g = L_beta(:,g);
            L_g =  Ls(:,:,g);
            y_minus_other = (1/(2*n))*(Y - sum(L_beta(:,setdiff(1:m,g)),2));
            L_g_y_minus = L_g'*y_minus_other;
            grad_g = -L_g_y_minus + (1/(2*n))*L_g'*L_beta_g;
            if norm(L_g_y_minus,2) <= lambda/m
                d_g = -Beta_g;
            else
                grad_minus_h = -grad_g + h_all(g)*Beta_g;
                d_g = 1/h_all(g)*(-grad_g - (lambda/m)*grad_minus_h/norm(grad_minus_h,2));
            end
            
            Beta(:,g) = Beta_g + 1*d_g;
            L_beta(:,g) = L_g*Beta(:,g);
        end

        currObj = objective(Beta);
        currTime = cputime - startTime;
        objHistory = [objHistory; currObj];
        timeHistory = [timeHistory; currTime];

        if params.optVerbose && mod(iter, params.optVerbosePerIter) == 0
            fprintf('bcgdDiagHessian #%d (%.4f): currObj: %0.5e\n\n', ...
                iter, currTime, currObj);
        end
        if abs( (currObj - prevObj) / currObj ) < params.tolerance
            break;
        end

        % Update
        prevObj = currObj;
    end

    % statistics
    optStats.objective = objHistory;
    optStats.time = timeHistory;
    optBeta = Beta;
end
