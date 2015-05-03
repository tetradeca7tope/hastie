function decomposition = obtainGroups(numDims, decomposition)
% A function to obtain the groups

  switch decomposition.setting 

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
        % To add all 1D components or not.
        if ~isfield(decomposition, 'addAll1DComps') |decomposition.addAll1DComps
          groups = num2cell((1:numDims)', 2); % Add all 1D components
          numAdditionalGroups = numRandGroups - numDims;
        else
          groups = cell(0, 1);
          numAdditionalGroups = numRandGroups;
        end

        % now act depending on whether nchoosek is large
        if nchoosek(numDims, groupSize) * groupSize < 10000
          % If so, just enumerate all combinations and pick out the top
          % numAdditional Groups
          allCombs = (combnk(1:numDims, groupSize));
              if size(allCombs, 1) < numAdditionalGroups
                error('M is too large for group size !');
              end
          shuffleOrder = randperm(size(allCombs,1))';
          selGroupIdxs = sort( shuffleOrder(1:numAdditionalGroups, :) );
          randGroups = allCombs(selGroupIdxs, :);
        else
          numAdditionalGroupsUB = ceil(1.5*numAdditionalGroups);
          randGroups = zeros(numAdditionalGroupsUB, groupSize);
          for j = 1:numAdditionalGroupsUB
            randGroups(j,:) = sort(randperm(numDims, groupSize));
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
        end
        % Finally add randGroups to groups
        count = numel(groups);
        for j = 1:size(randGroups, 1)
          count = count + 1;
          groups{count} = randGroups(j,:);
        end 

      end

    case 'groups'
    % In this case, the groups are given and there is nothing to do.
      groups = decomposition.groups;

    otherwise
      errStr = sprintf('Unknown Decomposition setting: %s\n', ...
        decomposition.setting);
      error(errStr);

  end

  decomposition.groups = groups;

end


