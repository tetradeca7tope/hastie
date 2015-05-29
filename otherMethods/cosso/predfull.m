function output = predfull(x, y, xnew, folds)
     [n,d] = size(x);
     [nnew,d] = size(xnew);
     D = d * (d+1)/2;
     est = cvfull(x, y, folds);
     
     theta = est(n+1, 1:D);
     b = est(n+1, D+1);
     c = est(1:n, D+1);

     Knew = zeros(nnew, n);
     for i = 1:d
         Knew = Knew + theta(i) * kernel(xnew(:,i), x(:,i));
     end
     index = d;
     for i = 1:(d-1)
        for j = (i+1):d
          index = index + 1;
          Knew = Knew + theta(index) * (kernel(xnew(:,i), x(:,i)) .* kernel(xnew(:,j), x(:,j)));
        end
     end

     prediction = Knew * c + b;
     output = [prediction; theta'];
