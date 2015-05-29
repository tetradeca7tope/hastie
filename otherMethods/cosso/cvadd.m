function output = cvadd(x, y, folds)

[n, d] = size(x);

K3darray = zeros(n,n,d);
for i = 1:d
   K3darray(:,:,i) = kernel(x(:,i), x(:,i));
end

splitsize = floor(n/folds) * ones(folds, 1);
splitsize(folds) = n - (folds - 1) * floor(n/folds);
p = randperm(n);

theta = ones(d,1);
Kth0 = zeros(n,n);
for i = 1:d
   Kth0 = Kth0 + theta(i) * K3darray(:,:, i);
end

lambda0 = cvlambda0(Kth0, y, p, folds, splitsize);

m = 20;
a = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 30, 35]; 

cv = zeros(m,folds);

for run = 1:folds
   tuneindex = p(((run - 1) * floor(n/folds) + 1) : ((run - 1) * floor(n/folds) + splitsize(run)));
   temp = zeros(n, 1);
   temp(tuneindex) = 1;
   ytune = y(logical(temp));
   ytrain = y(logical(1-temp));

   ntrain = length(ytrain);

   K3dtrain = zeros(ntrain,ntrain,d);
   for i = 1:d
      K3dtrain(:,:,i) = K3darray(logical(1-temp), logical(1-temp), i);
   end

   theta = ones(d,1);
   Ktrainth0 = zeros(ntrain,ntrain);
   for i = 1:d
      Ktrainth0 = Ktrainth0 + theta(i) * K3dtrain(:,:, i);
   end

   bigKtrainth0 = [Ktrainth0 + lambda0 * eye(ntrain), ones(ntrain,1); ones(1,ntrain), 0];

   cb0 = bigKtrainth0 \ [ytrain;0];
   c0 = cb0(1:ntrain);
   b0 = cb0(ntrain+1);
   G0 = zeros(ntrain,d);
   for inneriter = 1:d
      G0(:, inneriter) = K3dtrain(:,:,inneriter) * c0;
   end
   for iter = 1:m
      M = a(iter);
      solutiontrain = twostepadd(K3dtrain, ytrain, d, Ktrainth0, G0, c0, b0, lambda0, M);
      Kpred = zeros(n - ntrain, ntrain);
      for inneriter = 1:d
         Kpred = Kpred + solutiontrain(ntrain+1, inneriter) * K3darray(logical(temp), logical(1-temp),inneriter);
      end
      prediction = Kpred * solutiontrain(1:ntrain, d+1) + solutiontrain(ntrain+1, d+1);
      cv(iter, run) = (norm(ytune - prediction))^2;
   end
end

meancv = cv * ones(folds, 1) / n;
[C, cviter] = min(meancv);

bigKth0 = [Kth0 + lambda0 * eye(n), ones(n,1); ones(1,n), 0];

cb0 = bigKth0 \ [y;0];
c0 = cb0(1:n);
b0 = cb0(n+1);
G0 = zeros(n,d);
for inneriter = 1:d
   G0(:, inneriter) = K3darray(:,:,inneriter) * c0;
end

   M = a(cviter);
   solution = twostepadd(K3darray, y, d, Kth0, G0, c0, b0, lambda0, M);
output = [solution, [meancv; zeros(n+1-m, 1)]]; 




























