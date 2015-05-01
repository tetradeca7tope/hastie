function [optBeta, optStats] = bcdExact(Ls, Y, lambda, params)
% Uses trust-region Newton to solve each group exactly
% Efficient Block-coordinate Descent Algorithms for the Group Lasso
% http://www.optimization-online.org/DB_FILE/2010/11/2806.pdf

    [n, ~, m] = size(Ls);
    % Create a function handle for the objective
    objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
    smoothObjGrad = @(arg) computeSmoothObjGradBeta(arg, Ls, Y);
    % Obtain optimisation params
    params = processOptParamsCommon(params, n, m);
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

    % Compute eigenvalues and eigenvectors of Hessian
    M_all = zeros(n, n, m);
    evectors_all = zeros(n,n,m);
    evalues_all = zeros(n,m);
    for g=1:m
        L_g = Ls(:,:,g);
        M_g = 1/(n)*L_g'*L_g;
        [V, D] = eig(M_g);
        evectors_all(:,:,g) = V; % each column is an evector
        evalues_all(:,g) = diag(D);
        assert(min(diag(D))>0);
        M_all(:,:,g) = M_g;
    end

    for iter = 1:params.maxNumIters
        for g=1:m
            L_g = Ls(:,:,g);
            y_minus_other = (1/n)*(Y - sum(L_beta(:,setdiff(1:m,g)),2));
            L_g_y_minus_other = L_g'*y_minus_other;
            newbeta = zeros(n,1);
            if norm(L_g_y_minus_other,2) > lambda/m
                Delta = newton_trust(-L_g_y_minus_other, ...
                    evectors_all(:,:,g), evalues_all(:,g), lambda/m);
                Lhs = Delta*M_all(:,:,g)+(lambda/m)*eye(n);
                yy = Lhs\L_g_y_minus_other;
                newbeta = Delta*yy;
            end
            if ~isequal(Beta(:,g), newbeta)
                obj_before = objective(Beta);
                oldbeta = Beta(:,g);
                Beta(:,g) = newbeta;
                L_beta(:,g) = L_g*newbeta;
                obj_after = objective(Beta);
                if obj_after > obj_before
                    %fprintf('before:%f after:%f normbefore:%f normafter:%f\n', ...
                    %    obj_before, obj_after, norm(oldbeta), norm(newbeta));
                end
                fprintf('iter=%i g=%i obj_after=%4.5e\n', iter, g, obj_after);
            end
        end

        currObj = objective(Beta);
        currTime = cputime - startTime;
        objHistory = [objHistory; currObj];
        timeHistory = [timeHistory; currTime];

        if params.optVerbose && mod(iter, params.optVerbosePerIter) == 0
            fprintf('bcdExact #%d (%.4f): currObj: %0.5e\n\n', ...
                iter, currTime, obj);
        end
        
        if abs( (currObj - prevObj) / currObj ) < params.tolerance
            %break;
        end

        % Update
        prevObj = currObj;

    end

    % statistics
    optStats.objective = objHistory;
    optStats.time = timeHistory;
    optBeta = Beta;

end

function [Delta] = newton_trust(p_g, evectors, evalues, lambda)
    max_trust_iters = 10;
    trust_tol = 1.0e-3;
    n = size(p_g, 1);
    qtp2 = zeros(n,1);
    for i=1:n
        qtp2(i) = (evectors(:,i)'*p_g)^2;
    end
    fun = @(x) 1 - 1/sqrt(sum(qtp2 ./ ((evalues*x+lambda).^2)));
    Delta = fzero(fun, 0);
    assert(Delta > 0);
    return; % TODO
    success = 0;
    Delta = 0;
    for iter=1:max_trust_iters
        gamma_delta_lambda = evalues*Delta+lambda;
        left_sum = sum(qtp2 ./ (gamma_delta_lambda.^2));
        right_sum = sum((qtp2 .* evalues) ./ (gamma_delta_lambda.^3));
        f = 1 - 1/sqrt(left_sum);
        fprime = -1 * power(left_sum,-3/2) * right_sum;
        %f_plus_h = 1 - 1/sqrt(sum(qtp2 ./ ((evalues*(Delta+1e-6)+lambda).^2)));
        %fprime_diff = (f_plus_h - f)/1e-6;
        %fprintf('f:%f fprime:%f fprime_diff:%f \n', f, fprime, fprime_diff);
        Delta = Delta - f/fprime;
        if abs(f)<trust_tol
            success = 1;
            break;
        end
    end
    assert(success==1);
    assert(Delta > 0);
end
