% example 1 in the paper.
rand('state',1000);
n=100;
ntest = 10000;
d = 10;
noiselevel = 1.3175;

xall = rand((n+ntest),d); %generate all data points, both training and testing.

ftrueall = 5 * f1(xall(:,1)) + 3 * f2(xall(:,2)) + 4 * f3(xall(:,3)) + 6 * f4(xall(:,4));

xtrain = xall(1:n,:);          
xtest = xall((n+1):(n+ntest), :);
ftrain = ftrueall(1:n);
ftest = ftrueall((n+1):(n+ntest));
ytrain = ftrain + randn(n,1) * noiselevel;

predtestandtheta = predadd(xtrain, ytrain, xtest, 5); %given xtrain, ytrain, apply additive cosso with 5 fold cross validation, and make predictions at xtest.
predtest = predtestandtheta(1:ntest, 1);

theta = predtestandtheta((ntest+1):(ntest+d), 1)
mse = (norm(ftest - predtest))^2 / ntest

%circuit example. estimating the impedance Z.
rand('state',1000);
n=100;
ntest = 10000;
d = 4;
D = d * (d+1)/2;
noiselevel = 125;

xall = rand((n+ntest),d); %generate all data points, both training and testing.

ftrueall = ((100 * xall(:,1)).^2 + (2 * pi * (20 + 260 * xall(:,2)) .* xall(:,3) - (2 * pi * (20 + 260 * xall(:,2)) .* (xall(:,4) * 10 + 1)).^(-1) ).^2 ).^(0.5);

xtrain = xall(1:n,:);
xtest = xall((n+1):(n+ntest), :);
ftrain = ftrueall(1:n);
ftest = ftrueall((n+1):(n+ntest));
ytrain = ftrain + randn(n,1) * noiselevel;

predtestandtheta = predfull(xtrain, ytrain, xtest, 5); %given xtrain, ytrain, apply 2-way interaction cosso with 5 fold cross validation, and make predictions at xtest.
predtest = predtestandtheta(1:ntest, 1);

theta = predtestandtheta((ntest+1):(ntest+D), 1)
mse = (norm(ftest - predtest))^2 / ntest





