function decomposition = obtainGroups(numDims, decomposition)
% A function to obtain the groups

  setting = decomposition.setting;

  switch setting

    case 'groupSize'
    % Creates all groups of given groupSize
      groupSize = decomposition.groupSize;
      groups = num2cell( combnk(1:numDims, groupSize), 2);
    
    case 'maxGroupSize'
    % Creates all groups of size up to maxGroupSize
      groups = num2cell( combnk(1:numDims, decomposition.maxGroupSize), 2);
      count = numel(groups);
      for j = (decomposition.maxGroupSize-1):(-1):1
        newGroups =  combnk(1:numDims, j);
        for k = 1:size(newGroups, 1)
          count = count + 1;
          groups{count} = newGroups(k,:);
        end
      end

    case 'randomGroups'
    % Randomly creates decomposition.numRandGroups of size groupSize
      numRandGroups = decomposition.numRandGroups;
      groupSize = decomposition.groupSize;
      if decomposition.numRandGroups > nchoosek(numDims, groupSize);
        decomposition.numRandGroups = nchoosek(numDims, groupSize);
        groups = num2cell( combnk(1:numDims, groupSize), 2);
      else
        groups = num2cell( (1:numDims)', 2); % Add all 1D components
        numAdditionalGroups = numRandGroups - numDims;
        numAdditionalGroupsUB = ceil(1.1*numAdditionalGroups);
        randGroups = zeros(numAdditionalGroupsUB, groupSize);
        for j = 1:numAdditionalGroupsUB
          randGroups(j,:) = randperm(numDims, groupSize);
        end
        randGroups = unique(randGroups, 'rows');
        % Just shuffle them again.
        randGroups = randGroups( randperm(size(randGroups, 1)), :);
        if size(randGroups, 1) > numAdditionalGroups,
          randGroups = randGroups(1:numAdditionalGroups, :);
        else
          numAdditionalGroups = size(randGroups, 1);
          numRandGroups = numAdditionalGroups + numDims;
          decomposition.numRandGroups = numRandGroups;
        end
        % Now add them all to groups
        randGroups = unique(randGroups, 'rows'); % Just sorting them
        count = numel(groups);
        for j = 1:size(randGroups, 1)
          count = count + 1;
          groups{count} = randGroups(j,:);
        end 
      end

    case 'groups'
    % In this case, the groups are given and there is nothing to do.
      groups = decomposition.groups;

  end

  decomposition.groups = groups;

end


