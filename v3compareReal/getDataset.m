function [Xtr, Ytr, Xte, Yte] = getDataset(dataset)

  rng('default');

  switch dataset

    case 'debug'
      Xtr = rand(10,4);
      Xte = rand(10,4);
      Ytr = rand(10,1);
      Yte = rand(10,1);

    case 'parkinson21'
      L = load('datasets/parkinson21.txt');
      L = shuffleData(L);
      attrs = [1:14, 20:26]; label = 15;
      trIdxs = (1:520)';
      teIdxs = (521:1040)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'parkinson21-small'
      L = load('datasets/parkinson21.txt');
      L = shuffleData(L);
      attrs = [1:14, 20:26]; label = 15;
      trIdxs = (1:220)';
      teIdxs = (221:440)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'propulsion'
      L = load('datasets/propulsion.txt');
%         L = L(1:2000, :); L = shuffleData(L);
%         trIdxs = (1:1000)'; teIdxs = (1001:2000)';
        L = L(1:400, :); L = shuffleData(L);
        trIdxs = (1:200)'; teIdxs = (201:400)';
      attrs = [1 3:8 10:18]; label = 2;
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'housing'
      L = load('datasets/housing.txt');
      L = shuffleData(L);
      attrs = [2:3 5:14];  label = 1;
      trIdxs = (1:256)'; teIdxs = (257:506)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'forestfires'
      L = load('datasets/forestfires.txt');
      L = shuffleData(L);
      trIdxs = (140:350)';
      teIdxs = (1:517)';
      attrs = [1:10]; label = 11;
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      
    

    otherwise
      error('Unknown Dataset');

  end



end

% Another utility function
function [Xtr, Ytr, Xte, Yte] = ...
  partitionData(L, attrs, label, trainIdxs, testIdxs)
  Xtr = L(trainIdxs, attrs);
  Ytr = L(trainIdxs, label);
  Xte = L(testIdxs, attrs);
  Yte = L(testIdxs, label);

  % Now normalize the dataset to have unit variance in input axis
  meanXtr = mean(Xtr);
  stdXtr = std(Xtr);
  meanYtr = mean(Ytr);
  stdYtr = std(Ytr);

  % process files
  function X = normalizeX(X)
    X = bsxfun(@rdivide, bsxfun(@minus, X, meanXtr), stdXtr); 
  end
  function Y = normalizeY(Y)
    Y = (Y - meanYtr)/stdYtr;
  end

  Xtr = normalizeX(Xtr);
  Xte = normalizeX(Xte);
  Ytr = normalizeY(Ytr);
  Yte = normalizeY(Yte);
end


% Shuffles the data
function [X, Y] = shuffleData(X, Y)
  n = size(X, 1);
  shuffleOrder = randperm(n);
  X = X(shuffleOrder, :);
  if nargin > 1
    Y = Y(shuffleOrder, :);
  end
end

