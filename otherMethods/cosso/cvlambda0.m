function lambda0 = cvlambda0(K, y, p, folds, splitsize)

% tune lambda0.

n = length(y);
m = 25;
cvlambda = zeros(m,folds);

for run = 1:folds
   tuneindex = p(((run - 1) * floor(n/folds) + 1) : ((run - 1) * floor(n/folds) + splitsize(run)));
   temp = zeros(n, 1);
   temp(tuneindex) = 1;
   ytune = y(logical(temp));
   ytrain = y(logical(1-temp));

   ntrain = length(ytrain);
   Ktrain = K(logical(1-temp), logical(1-temp));
   for j = 1:m
      lambda0 = 2^( - j);

      bigKtrain = [Ktrain + lambda0 * eye(ntrain), ones(ntrain,1); ones(1,ntrain), 0];
      cb = bigKtrain\[ytrain;0];
      Kpred = K(logical(temp), logical(1-temp));
      prediction = Kpred * cb(1:ntrain, 1) + cb(ntrain + 1);
      cvlambda(j, run) = (norm(ytune - prediction))^2;
   end
end

meancvlambda = cvlambda * ones(folds, 1) / n;

[C, cviter] = min(meancvlambda);

lambda0 = 2^( - cviter);




