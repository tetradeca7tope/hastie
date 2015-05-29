function output = cvfull(x, y, folds)

[n, d] = size(x);
D = d * (d+1)/2;
K3darray = zeros(n,n,d);
for i = 1:d
   K3darray(:,:,i) = kernel(x(:,i), x(:,i));
end

splitsize = floor(n/folds) * ones(folds, 1);
splitsize(folds) = n - (folds - 1) * floor(n/folds);
p = randperm(n);

theta = ones(D,1);
Kth0 = zeros(n,n);
for i = 1:d
   Kth0 = Kth0 + theta(i) * K3darray(:,:, i);
end
index = d;
for i = 1:(d-1)
     for j = (i+1):d
        index = index + 1;
        Kth0 = Kth0 + theta(index) * (K3darray(:,:, i) .* K3darray(:,:, j));
     end
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

   Ktrainth0 = Kth0(logical(1-temp), logical(1-temp));

   bigKtrainth0 = [Ktrainth0 + lambda0 * eye(ntrain), ones(ntrain,1); ones(1,ntrain), 0];

   cb0 = bigKtrainth0 \ [ytrain;0];
   c0 = cb0(1:ntrain);
   b0 = cb0(ntrain+1);
   G0 = zeros(ntrain,D);
   for i = 1:d
      G0(:, i) = K3dtrain(:,:,i) * c0;
   end
   index = d;
   for i = 1:(d-1)
       for j = (i+1):d
           index = index + 1;
           G0(:, index) = (K3dtrain(:,:,i) .* K3dtrain(:,:,j)) * c0;
       end
   end

   for iter = 1:m
      M = a(iter);
      solutiontrain = twostepfull(K3dtrain, ytrain, d, Ktrainth0, G0, c0, b0, lambda0, M);
      theta = solutiontrain(ntrain+1, 1:D);
      Kpred = zeros(n - ntrain, ntrain);
      for i = 1:d
         Kpred = Kpred + theta(i) * K3darray(logical(temp), logical(1-temp),i);
      end
      index = d;
      for i = 1:(d-1)
          for j = (i+1):d
              index = index + 1;
              Kpred = Kpred + theta(index) * (K3darray(logical(temp), logical(1-temp),i) .* K3darray(logical(temp), logical(1-temp),j));
          end
      end
      prediction = Kpred * solutiontrain(1:ntrain, D+1) + solutiontrain(ntrain+1, D+1);
      cv(iter, run) = (norm(ytune - prediction))^2;
   end
end

meancv = cv * ones(folds, 1) / n;
[C, cviter] = min(meancv);

bigKth0 = [Kth0 + lambda0 * eye(n), ones(n,1); ones(1,n), 0];

cb0 = bigKth0 \ [y;0];
c0 = cb0(1:n);
b0 = cb0(n+1);
G0 = zeros(n,D);
for i = 1:d
   G0(:, i) = K3darray(:,:,i) * c0;
end
index = d;
for i = 1:(d-1)
    for j = (i+1):d
          index = index + 1;
          G0(:, index) = (K3darray(:,:,i) .* K3darray(:,:,j)) * c0;
    end
end

M = a(cviter);
solution = twostepfull(K3darray, y, d, Kth0, G0, c0, b0, lambda0, M);
output = [solution, [meancv; zeros(n+1-m, 1)]]; 
