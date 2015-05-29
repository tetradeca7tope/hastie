function output = predadd(x, y, xnew, folds)
     [n,d] = size(x);
     [nnew,d] = size(xnew);
     est = cvadd(x, y, folds);
     
     theta = est(n+1, 1:d);
     b = est(n+1, d+1);
     c = est(1:n, d+1);

     Knew = zeros(nnew, n);
     for i = 1:d
         Knew = Knew + theta(i) * kernel(xnew(:,i), x(:,i));
     end

     prediction = Knew * c + b;
     output = [prediction; theta'];
