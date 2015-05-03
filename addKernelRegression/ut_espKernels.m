% unit test for esp kernels

clear all;
close all;
clc;

numDims = 3;
n = 5;
m = 4;

X = rand(n,numDims);
Y = rand(m, numDims);

bws = rand(numDims, 1);

[~, allKs] = espKernels(X, X, bws),
[~, allKs] = espKernels(X, Y, bws),

