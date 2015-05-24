% Unit Test for obtainDecomposition.m and kernelSetup.m

% Uncomment printing of groups in obtainDecomposition.m.m (~ line 55)
X = rand(10,6); 
Y = sum(X.^2, 2) + X(:,1);
numDims = size(X, 2);

dc1.setting = 'groupSize';    dc1.groupSize    = 3; 
[kernelFunc, dc1] = kernelSetup(X, Y, dc1);
printGroups(dc1.groups);
[K, allKs] = kernelFunc(X, X); allKs(:,:,2:5), K,
fprintf('\n'); pause,

dc2.setting = 'maxGroupSize'; dc2.maxGroupSize = 3; 
[kernelFunc, dc2] = kernelSetup(X, Y, dc2);
printGroups(dc2.groups);
[K, allKs] = kernelFunc(X, X); allKs(:,:,2:5), K,
fprintf('\n'); pause,

dc3.setting = 'randomGroups'; dc3.groupSize    = 3; dc3.numRandGroups=20;
[kernelFunc, dc3] = kernelSetup(X, Y, dc3);
printGroups(dc3.groups);
[K, allKs] = kernelFunc(X, X); allKs, K,
fprintf('\n'); pause,

