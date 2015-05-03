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

