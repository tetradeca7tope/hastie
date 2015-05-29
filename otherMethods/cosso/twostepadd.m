function solution = twostepadd(K3d, y, d, Kth0, G0, c0, b0, lambda0, M)

     % Given the data, lambda0, and M, solve for the estimate. G0, c0, b0 is the starting G, c, b. The solution is arranged as a (n+1) by (d+1) matrix.

n = length(y);
theta = ones(d,1);
ycb0 = y - lambda0 * c0 /2 - b0;

warning off;
% theta = lsqlin(G0, ycb0, ones(1,d), M, [], [], zeros(d,1));
options.Display = 'off';
theta = lsqlin(G0, ycb0, ones(1,d), M, [], [], zeros(d,1), [], [], options);

G = G0;

   Kth = zeros(n,n);
   for inneriter = 1:d
      Kth = Kth + theta(inneriter) * K3d(:,:, inneriter);
   end

   bigKth = [Kth + lambda0 * eye(n), ones(n,1); ones(1,n), 0];
   cb = bigKth \ [y;0];
   c = cb(1:n);
   b = cb(n+1);
   for inneriter = 1:d
      G(:, inneriter) = K3d(:,:,inneriter) * c;
   end

   solution = [[G; theta'], [c; b]];







