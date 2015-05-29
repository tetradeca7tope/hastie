
clear all;
dirName = 'results';
dirSearchName = sprintf('%s/v2-*.mat', dirName);
files = dir(dirSearchName);

% Now add all of it
for i = 1:numel(files)
  fileName = sprintf('%s/%s', dirName, files(i).name);
  load(fileName);
%   saveFileName = fi;
  plotV2ResultsDuplicate;
  pause;

  if exist('numDims', 'var')
    numDims,
  end

end

