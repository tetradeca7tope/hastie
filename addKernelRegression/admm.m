function [optBeta, optStats] = admm(Ls, Y, lambda, params)
    [n, ~, m] = size(Ls);
    % Create a function handle for the objective
    objective = @(arg) computeObjBeta(reshape(arg, [n m]), Ls, Y, lambda);
    % Obtain optimisation params
    params = processOptParamsCommon(params, n, m);
    rho = params.rho;
    alpha = params.alpha;

    b = Y/sqrt(n);
    A = zeros(n, n*m);
    for j=0:m-1
        A(:,j*n+1:(j+1)*n) = Ls(:,:,j+1)/sqrt(n);
    end
    p = n*ones(m,1);
    [x, z, history] = group_lasso_admm(A, b, lambda/m, p, params, objective);
    optBeta = reshape(x, [n m]);
    optStats.objective = history.objval';
    optStats.time = history.time;

end

function [x, z, history] = group_lasso_admm(A, b, lambda, p, params, objFunc)
% group_lasso  Solve group lasso problem via ADMM
%
% [x, history] = group_lasso(A, b, p, lambda, rho, alpha);
% 
% solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
%
% The input p is a K-element vector giving the block sizes n_i, so that x_i
% is in R^{n_i}.
% 
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

%% Global constants and defaults
rho = params.rho;
alpha = params.alpha;
QUIET    = 0;
MAX_ITER = params.maxNumIters;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
startTime = cputime;
timeHistory = 0;
%% Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;
% check that sum(p) = total number of elements in x
if (sum(p) ~= n)
    error('invalid partition');
end

% cumulative partition
cum_part = cumsum(p);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

% pre-factor
[L U] = factor(A, rho);
history.objval = objFunc(x);

for k = 1:MAX_ITER

    % x-update
    q = Atb + rho*(z - u);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    % z-update
    zold = z;
    start_ind = 1;
    x_hat = alpha*x + (1-alpha)*zold;
    for i = 1:length(p),
        sel = start_ind:cum_part(i);
        z(sel) = shrinkage(x_hat(sel) + u(sel), lambda/rho);
        start_ind = cum_part(i) + 1;
    end
    u = u + (x_hat - z);
    
    % diagnostics, reporting, termination checks
    history.objval(k+1)  = objFunc(x);
    
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    currTime = cputime - startTime;
    timeHistory = [timeHistory; currTime];
    if params.optVerbose & mod(k, params.optVerbosePerIter) == 0
      fprintf('ADMM #%d (%0.3f): currObj: %.4e\n', ...
        k, currTime, history.objval(k+1));
    end
end
    history.time = timeHistory;

end

function p = objective(A, b, lambda, cum_part, x, z)
    obj = 0;
    start_ind = 1;
    for i = 1:length(cum_part),
        sel = start_ind:cum_part(i);
        obj = obj + norm(z(sel));
        start_ind = cum_part(i) + 1;
    end
    p = ( 1/2*sum((A*x - b).^2) + lambda*obj );
end

function z = shrinkage(x, kappa)
    z = pos(1 - kappa/norm(x))*x;
end

function [L U] = factor(A, rho)
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

