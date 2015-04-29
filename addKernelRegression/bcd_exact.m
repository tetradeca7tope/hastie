function [all_alphas, all_betas, all_stats] = bcd_exact(y, L_all, lambdas)
% Uses trust-region Newton to solve each group exactly
% Efficient Block-coordinate Descent Algorithms for the Group Lasso
% http://www.optimization-online.org/DB_FILE/2010/11/2806.pdf
% y(n,1)
% L_all(n,n,M)
% lambdas(k,1) for warm-starting with k warm-starts
% alphas(n,M,k) 
% betas(n,M,k)

    [n, ~, m] = size(L_all);
    k = size(lambdas(:),1);
    lambdas = sort(lambdas,'descend');
    %params = processOptParamsCommon(params, n, m);
    %max_iters = params.maxNumIters;
    %f_tol = params.tolerance;
    max_iters = 10;
    f_tol = 1.0e-4;

    %alphas = params.initAlpha; % zeros(n, m);
    %alphas = zeros(n,m);
    betas = zeros(n,m);
    L_betas = zeros(n,m);
    for g=1:m
        L_betas(:,g) = L_all(:,:,g)*betas(:,g);
    end

    % Saved outputs for all warm-starts
    all_alphas = zeros(n,m,k);
    all_betas = zeros(n,m,k);
    all_stats = [];
    tic; startTime = cputime;

    % Compute eigenvalues and eigenvectors of Hessian
    M_all = zeros(n, n, m);
    evectors_all = zeros(n,n,m);
    evalues_all = zeros(n,m);
    for g=1:m
        L_g = L_all(:,:,g);
        M_g = 1/(2*n)*L_g'*L_g;
        [V, D] = eig(M_g);
        evectors_all(:,:,g) = V; % each column is an evector
        evalues_all(:,g) = diag(D);
        assert(min(diag(D))>0);
        M_all(:,:,g) = M_g;
    end
    
    for warm_iter = 1:k
        lambda = lambdas(warm_iter)/m;
        obj_history = objective_betas(y, L_betas, lambda, betas);
        time_history = cputime - startTime;

        % Run block coordinate descent
        for iter=1:max_iters
            for g=randperm(m)
                L_g = L_all(:,:,g);
                y_minus_other = 1/(n)*(y - sum(L_betas(:,setdiff(1:m,g)),2));
                L_g_y_minus_other = L_g*y_minus_other;
                newbeta = zeros(n,1);
                if norm(L_g_y_minus_other,2) > lambda
                    Delta = newton_trust(-L_g_y_minus_other, ...
                        evectors_all(:,:,g), evalues_all(:,g), lambda);
                    Lhs = Delta*M_all(:,:,g)+lambda*eye(n);
                    yy = Lhs\L_g_y_minus_other;
                    newbeta = Delta*yy;
                end
                if ~isequal(betas(:,g), newbeta)
                    betas(:,g) = newbeta;
                    L_betas(:,g) = L_g*newbeta;
                    obj_after = objective_betas(y,L_betas,lambda,betas)
                    fprintf('iter=%i g=%i obj_after=%4.5e\n', iter, g, obj_after);
                end
            end

            obj = objective_betas(y, L_betas, lambda, betas);
            currTime = cputime - startTime;
            obj_history = [obj_history; obj];
            time_history = [time_history; currTime];

            %if params.verbose & mod(iter, params.verbosePerIter) == 0
              fprintf('#%d (%.4f): currObj: %0.5e\n', ...
                iter, currTime, obj);
            %end
            
            if abs((obj-obj_history(end-1))/obj_history(end-1)) <= f_tol
                fprintf('Terminating BCD for lambda=%f at %d iterations.\n', ...
                    lambda, iter);
                break;
            end
        end

        % Compute alphas from betas and save everything
        all_betas(:,:,warm_iter) = betas;
        for g=1:m
            all_alphas(:,g,warm_iter) = (L_all(:,:,g)')\betas(:,g);
        end
        all_stats(warm_iter).objective = obj_history;
        all_stats(warm_iter).time = time_history;
    end
end

function [Delta] = newton_trust(p_g, evectors, evalues, lambda)
    max_trust_iters = 8;
    trust_tol = 1.0e-5;
    n = size(p_g, 1);
    qtp2 = zeros(n,1);
    for i=1:n
        qtp2(i) = (evectors(:,i)'*p_g)^2;
    end
    %fun = @(x) 1 - 1/sqrt(sum(qtp2 ./ ((evalues*x+lambda).^2)));
    %Delta = fzero(fun, 0)
    %assert(Delta > 0);
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

function [obj] = objective_betas(y, L_betas, lambda, betas)
    [n, m] = size(L_betas);
    fit = y - sum(L_betas,2);
    obj_fit = 1/(2*n)*fit'*fit;
    l12norm = 0;
    for g=1:m
        l12norm = l12norm + norm(betas(:,g),2);
    end
    obj = obj_fit + lambda/m*l12norm;
end
