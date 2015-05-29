function solution = twostepfull(K3d, y, d, Kth0, G0, c0, b0, lambda0, M)

     % Given the data, lambda0, and M, solve for the estimate. G0, c0, b0 is the starting G, c, b. The solution is arranged as a (n+1) by (D+1) matrix.

n = length(y);
D = d * (d+1) /2;
ycb0 = y - lambda0 * c0 /2 - b0;

warning off;
theta = lsqlin(G0, ycb0, ones(1,D), M, [], [], zeros(D,1));

G = G0;

Kth = zeros(n,n);
for i = 1:d
   Kth = Kth + theta(i) * K3d(:,:, i);
end
index = d;
for i = 1:(d-1)
     for j = (i+1):d
        index = index + 1;
        Kth = Kth + theta(index) * (K3d(:,:, i) .* K3d(:,:, j));
     end
end

bigKth = [Kth + lambda0 * eye(n), ones(n,1); ones(1,n), 0];
cb = bigKth \ [y;0];
c = cb(1:n);
b = cb(n+1);
for i = 1:d
   G(:, i) = K3d(:,:,i) * c;
end
index = d;
for i = 1:(d-1)
    for j = (i+1):d
           index = index + 1;
           G(:, index) = (K3d(:,:,i) .* K3d(:,:,j)) * c;
    end
end

solution = [[G; theta'], [c; b]];




