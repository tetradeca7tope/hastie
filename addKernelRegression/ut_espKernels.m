% unit test for esp kernels

clear all;
close all;
clc;

% numDims = 5; order = 3;
% numDims = 80; order = 3;
numDims = 80; order = 40;
n = 5;
m = 4;

X = rand(n,numDims);
Y = rand(m, numDims);

bws = rand(numDims, 1);

[K, allKs] = espKernels(X, X, bws, order), 
pause,
[K, allKs] = espKernels(X, Y, bws, order),

