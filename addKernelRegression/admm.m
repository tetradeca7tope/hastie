function [optBeta, optStats] = admm(Ls, Y, lambda, params)
    [n, ~, m] = size(Ls);
    % Create a function handle for the objective
    objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
    % Obtain optimisation params
    params = processOptParamsCommon(params, n, m);
    rho = params.rho;
    alpha = params.alpha;

    L_all = Ls/sqrt(n);
    y = Y/sqrt(n);
    Lambda = lambda/m;
    A = zeros(n,n*m);
    for g=0:m-1
        A(:,g*n+1:(g+1)*n) = L_all(:,:,g+1);
    end
    Aty = A'*y;
    Lty = zeros(n,m);
    for g=1:m
        Lty(:,g) = L_all(:,:,g)'*y;
    end

    Beta = params.initBeta;
    Z = params.initZ;
    U = params.initU;
    beta = Beta(:);
    z = Z(:);
    u = U(:);

    % Set up book keeping
    prevObj = objective(Beta);
    objHistory = prevObj;
    timeHistory = 0;
    startTime = cputime;

    size(A)
    [LU_L, LU_U] = factor(A, rho);

    for iter = 1:params.maxNumIters

        % x-update
        q = Aty + rho*(z - u);    % temporary value
        if( m >= n )    % if skinny
            size(LU_U)
            size(LU_L)
            size(LU_L)
            beta = LU_U \ (LU_L \ q);
        else            % if fat
            beta = q/rho - (A'*(LU_U \ ( LU_L \ (A*q) )))/rho^2;
        end
        Beta = reshape(beta, [n m]);
        beta = Beta(:);

        % z-update
        Zold = Z;
        start_ind = 1;
        Beta_hat = alpha*Beta + (1-alpha)*Zold;
        for g = 1:m
            Z(:,g) = shrinkage(Beta_hat(:,g) + U(:,g), lambda/rho);
        end
        z = Z(:);

        % u-update
        U = U + (Beta_hat - Z);
        u = U(:);
        currObj = objective(Beta)
    end

end
function z = shrinkage(x, kappa)
    z = pos(1 - kappa/norm(x))*x;
end

function [L, U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
        L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
        L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end
