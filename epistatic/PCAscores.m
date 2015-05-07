function [scores] = PCAscores(A,k);

%
% [scores] = PCAscores(A,k);
%
% This function computes the scores used to identify PCA-correlated SNPs.
% It takes as input the SNP matrix A, properly encoded (see paper), and
% with the missing entries filled in. It also takes as input the number of
% significant principal components k; if k is set to zero, then our code
% automatically infers the number of significant principal components.
%
% The function returns the vector scores that contains a value for each
% SNP. SNPs corresponding to higher values are typically the most
% informative ones for reproducing the population structure.
%

if k == 0, % automatically detect the value of k
    k = estimate_k(A);
end

[U,S,V] = svd(A*A'); % compute the SVD of A times A'; this speeds up the SVD computation,
% assuming that we are given fewer individuals than SNPs

RSV = inv(sqrt(S(1:k,1:k)))*U(:,1:k)'*A; % compute the top k right singular vectors of A

if k == 1, % special case
    scores = RSV.^2; % compute the scores
else
    scores = sum(RSV.^2); % compute the scores
end

