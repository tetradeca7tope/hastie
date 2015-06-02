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
        L = shuffleData(L); L(:,2) = log(L(:,2));
%         L = L(1:2000, :); trIdxs = (1:1000)'; teIdxs = (1001:2000)';
        L = L(1:400, :); trIdxs = (1:200)'; teIdxs = (201:400)';
      attrs = [3:8 10:18]; label = 2;
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'housing'
      L = load('datasets/housing.txt');
      L = shuffleData(L);
      attrs = [2:3 5:14];  label = 1;
      trIdxs = (1:256)'; teIdxs = (257:506)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'music'
      L = load('datasets/music.txt');
      L = shuffleData(L);
      attrs = (2:91)'; label = 1;
      trIdxs = (1:1000)'; teIdxs = (1001:2000)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'music-small'
      L = load('datasets/music.txt');
      L = shuffleData(L);
      attrs = (2:91)'; label = 1;
      trIdxs = (1:400)'; teIdxs = (401:800)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'telemonitoring-total'
      L = load('datasets/telemonitoring.txt');
      L = L( L(:,3) == 0, :); % only select female candidates
      L = shuffleData(L);
      attrs = [2 4:5 7:22]; label = 6;
      trIdxs = (1:1000)'; teIdxs = (1001:1867)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'telemonitoring-total-small'
      L = load('datasets/telemonitoring.txt');
      L = L( L(:,3) == 0, :); % only select female candidates
      L = shuffleData(L);
      attrs = [2 4:5 7:22]; label = 6;
      trIdxs = (1:300)'; teIdxs = (301:600)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'telemonitoring-motor-small'
      L = load('datasets/telemonitoring.txt');
      L = L( L(:,3) == 0, :); % only select female candidates
      L = shuffleData(L);
      attrs = [2 4 5 7:22]; label = 5;
%       attrs = [2 4 7:22]; label = 5;
      trIdxs = (1:300)'; teIdxs = (301:600)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'forestfires'
      L = load('datasets/forestfires.txt');
      L = shuffleData(L);
      L(:,11) = log(L(:,11) + 1);
      trIdxs = (140:350)';
      teIdxs = (351:517)';
      label = 7; attrs = setdiff([1:11], label);
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      
    case 'blog'
      L = load('datasets/blog.txt');
      L = shuffleData(L);
      rmIdxs = (mean(abs(L)) < .11)';
      L = L(:, ~rmIdxs);
      trIdxs = (1:700)'; teIdxs = (701:1388)';
      label = 92; attrs = (1:91)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
    
    case 'LRGs'
      load('datasets/lrgReg.mat');
      Y = Y / std(Y);
      trLabels = 1:2000;
      teLabels = 2001:4000;
      Xtr = X(trLabels, :);
      Ytr = Y(trLabels, :);
      Xte = X(teLabels, :);
      Yte = Y(teLabels, :);

    case 'Skillcraft'
      L = load('datasets/skillcraft.txt');
      L = shuffleData(L);
      attrs = [2:15 17:20]; label = 16;
      trIdxs = (1:1700)'; teIdxs = (1701:3330)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      
    case 'Skillcraft-small'
      L = load('datasets/skillcraft.txt');
      L = shuffleData(L);
%       attrs = [2:15 17:20]; label = 16;
%       attrs = [2:16 18:20]; label = 17;
%       attrs = [2:17 19:20]; label = 18; % **
%       attrs = [2:18 20]; label = 19; 
      attrs = [2:19]; label = 20; % **
      trIdxs = (1:300)'; teIdxs = (1701:2000)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'Airfoil*'
      L = load('datasets/airfoil.txt');
      L = shuffleData(L);
      attrs = [1:5]; label = 6;
      trIdxs = (1:750)'; teIdxs = (751:1500)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      numAddDims = 36; % *
      Xtr = [Xtr, randn(size(Xtr,1), numAddDims)];
      Xte = [Xte, randn(size(Xte,1), numAddDims)];

    case 'CCPP*'
      load('datasets/ccpp.mat');
      L = [XTrain YTrain];
      L = shuffleData(L);
      attrs = [1:4]; label = 5;
      trIdxs = (1:2000)'; teIdxs = (2001:4000)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      numAddDims = 55; % *
      Xtr = [Xtr, randn(size(Xtr,1), numAddDims)];
      Xte = [Xte, randn(size(Xte,1), numAddDims)];

   case 'CCPP*small'     
      load('datasets/ccpp.mat');
      L = [XTrain YTrain];
      L = shuffleData(L);
      attrs = [1:4]; label = 5;
      trIdxs = (1:300)'; teIdxs = (2001:2300)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      numAddDims = 55; % *
      Xtr = [Xtr, randn(size(Xtr,1), numAddDims)];
      Xte = [Xte, randn(size(Xte,1), numAddDims)];
      
    case 'Insulin'
      load('datasets/epidata.mat');
      L = [insulin_data snp_data];
      L = shuffleData(L);
      attrs = [2:51]; label =1;
      trIdxs = (1:256)'; teIdxs = (257:506)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'Bleeding'
      load('datasets/bleed.mat');
      L = shuffleData(X1);
      preLabel = 1; preAttrs = [2: 1526];
      D = 100;
      Y = L(:, preLabel);
      X = L(:, preAttrs) * A(:,1:D);
      L = [Y X];
      attrs = [2:(D+1)]; label =1;
      trIdxs = (1:200)'; teIdxs = (201:351)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);

    case 'School'
      load('datasets/school.mat');
      L = [school_data_features school_data_output];
      L = shuffleData(L);
      label = 37; attrs = setdiff(1:36, label);
%       attrs = [1:36]; label =37;
      trIdxs = (1:90)'; teIdxs = (91:142)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);
      
    case 'Brain'
      load('datasets/brain.mat');
      L = zeros(0, 30);
      for i = 1:9
        L = [L; VoxelsShelter{i}];
      end
      L = shuffleData(L);
      label = 30; attrs = setdiff(1:30, label);
      trIdxs = (1:300)'; teIdxs = (301:540)';
      [Xtr, Ytr, Xte, Yte] = partitionData(L, attrs, label, trIdxs, teIdxs);


    case 'fMRI'
      load('datasets/fmri.mat');
      L = shuffleData(data);
      preLabel = 1; preAttrs = setdiff(1:37913, preLabel);
      D = 100;
      Y = L(:, preLabel);
      X = L(:, preAttrs) * A(:,1:D);
      L = [Y X];
      attrs = [2:(D+1)]; label =1;
      trIdxs = (1:700)'; teIdxs = (701:1351)';
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

