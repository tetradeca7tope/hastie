function [Xtr, Ytr, Xte, Yte] = getDataset(dataset)

  rng('default');

  switch dataset

    case 'parkinson21'
      L = load('parkinson21.txt');
      L = shuffleData(L);
      attrs = [1:14, 20:26]; label = 15;
      trIdxs = (1:520)';
      teIdxs = (521:1040)';
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

  Xtr = normalizeData(Xtr);
  Xte = normalizeData(Xte);
  Ytr = normalizeData(Ytr);
  Yte = normalizeData(Yte);
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

