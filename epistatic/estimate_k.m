function [k] = estimate_k(A);

%
% [k] = estimate_k(A);
%
% This function automatically estimates the number of significant principal
% components in the (properly encoded) SNP matrix A. It returns k, the
% estimate for the number of significant principal components. See the
% paper for more details.
%

CUTOFF = 1.15; % set a threshold
[m,n] = size(A); % compute the dimensions of A
[U,S,V] = svd(A*A'); % compute the SVD of A times A'; this speeds up our computations, assuming that we given more SNPs than individuals

k = 1; % initial value of k
while 1

    TA = U(:,k:size(U,2))*U(:,k:size(U,2))'*A; % discard the top k-1 principal components
    
    VectorA = reshape(TA,m*n,1); % randomly permute the entries of A
    tt = randperm(m*n); 
    Trand = reshape(VectorA(tt),m,n);
    shufflednorm = norm(Trand); % compute the norm of the permuted matrix
    
    if sqrt(S(k,k))/shufflednorm < CUTOFF % if there is no significantly more structure in TA when compared to a random permumation 
        k = k - 1;
        break;
    elseif k == min(m,n) % we are done
        break;
    else
        k = k + 1; % continue
    end
    
end