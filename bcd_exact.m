function [alphas, stats] = bcd_exact(y, K_all, lambda_1, lambda_2, lambda_3)
% Uses trust-region Newton to solve each group exactly
% Efficient Block-coordinate Descent Algorithms for the Group Lasso
% http://www.optimization-online.org/DB_FILE/2010/11/2806.pdf
    [n, ~, m] = size(K_all);
    max_iters = 100;
    f_tol = 1.0e-4;

    alphas = zeros(n, m);
    K_alphas = zeros(n,m);
    for g=1:m
        K_alphas(:,g) = K_all(:,:,g)*alphas(:,g);
    end

    obj_history = fast_objective(alphas,y,K_alphas,lambda_1,lambda_2,lambda_3)
    time_history = 0;
    tic;

    % Compute eigenvalues and eigenvectors of Hessian
    M_all = zeros(n, n, m);
    evectors_all = zeros(n,n,m);
    evalues_all = zeros(n,m);
    for g=1:m
        K_g = K_all(:,:,g);
        M_g = K_g'*K_g + lambda_1*K_g + lambda_2*eye(n,n);
        [V, D] = eig(M_g);
        evectors_all(:,:,g) = V; % each column is an evector
        evalues_all(:,g) = diag(D);
        assert(min(diag(D))>0);
        M_all(:,:,g) = M_g;
    end

    % Run block coordinate descent
    for iter=1:max_iters
        for g=randperm(m)
            K_g = K_all(:,:,g);
            y_minus_other = y - sum(K_alphas(:,setdiff(1:m,g)),2);
            K_g_y_minus_other = K_g*y_minus_other;
            newalpha = zeros(n,1);
            if norm(K_g_y_minus_other,2) > lambda_3
                Delta = newton_trust(-K_g_y_minus_other, ...
                    evectors_all(:,:,g), evalues_all(:,g), lambda_3);
                Lhs = Delta*M_all(:,:,g)+lambda_3*eye(n);
                yy = Lhs\K_g_y_minus_other;
                newalpha = Delta*yy;
            end
            if ~isequal(alphas(:,g), newalpha)
                alphas(:,g) = newalpha;
                K_alphas(:,g) = K_g*alphas(:,g);
                %obj_after = objective(alphas,y,K_all,lambda_1,lambda_2,lambda_3);
                %fprintf('iter=%i g=%i obj_after=%4.10f\n', iter, g, obj_after);
            end
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
