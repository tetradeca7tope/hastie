% Creates a series of bar charts

close all;
clear all;

allMethods = {'SALSA', 'SVR', 'RT', 'BF', 'MARS', 'RR'};
datasets = {'Galaxy', 'fMRI', 'Skillcraft', 'Telemonitoring', 'Housing'};

numDatasets = numel(datasets);

values = [0.00014, 0.15798, 0.02293, 0.94165, 0.00163, 0.13902;
          0.80730, 0.81376, 1.52834, 0.86197, 0.90850, 0.81005;
          0.54695, 0.66311, 1.08047, 0.83733, 0.54595, 0.70910;
          0.03473, 0.05246, 0.01376, 0.84412, 0.02400, 0.08053;
          0.26241, 0.38600, 1.06015, 0.64218, 0.42379, 9.60708];

for dsIter = 1:numDatasets

  figure;  hold on,
  currValues = values(dsIter, :);
  sortedValues = sort(currValues, 'ascend');
  height = min( 1.1*sortedValues(end), 10*sortedValues(2));

  [~, minIdx] = min(currValues);

  for j = 1:numel(currValues)
    h = bar(j, currValues(j));
    if j == minIdx,
      set(h, 'FaceColor', 'g');
    end
  end

  ylim([0, height]);
  xlim([0.4 6.6]);
  set(gca, 'XTickLabel', allMethods, 'FontSize', 17.2);
%   ylabel('Test MSE', 'rot', 0, 'Position', [-0.30 0.9*height]);
  ylabel('Test MSE', 'Position', [-0.32 0.5*height], 'FontSize', 19);
  set(gca, 'Position', [0.16 0.12 0.83 0.86], 'units', 'normalized');
  set(gcf, 'Position', [85 261 700 500]);
  box on,


end
