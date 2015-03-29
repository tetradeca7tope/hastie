% Unit tests for l2NormProxOp.m and groupSparsityProxOp.m

% l2NormProxOp
fprintf('l2NormProxOp\n');
x = (1:5)';
t = 1; l2NormProxOp(x, t),
t = 20; l2NormProxOp(x, t),


% Group Sparsity  
fprintf('Group Sparsity Prox Operator\n');
X = zeros(5, 4);
X(:,1) = (1:5)';
X(:,2) = (0.1:0.1:0.5)';
X(:,3) = (10:10:50)';
X(:,4) = 0.5*ones(5,1);
tVals = 0.75; groupSparsityProxOp(X, tVals),
tVals = 0.5:0.5:2; groupSparsityProxOp(X, tVals),

