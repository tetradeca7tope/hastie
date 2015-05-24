% Plots results out

plotColours = {'b', 'g', 'r', 'k', 'c', 'y', 'm', [255 128 0]/255, ...
  [76, 0, 153]/253, [102 102 0]/255};
plotColours = fliplr(plotColours);
% plotMarkers = {'o', '+', '*', 'x', 's', 'd', '^', 'p', '>', 'v'};
plotMarkers = {'o-', '+-', '*-', 'x-', 's-', 'd-', '^-', 'p-', '>-', 'v-'};
plotFunc = @loglog;

MS = 6;
LW = 2;

selOrderIdxs = [1 2 3 4 5 6];

errMeans = zeros(numOrderCands, numNCandidates);
errStds = zeros(numOrderCands, numNCandidates);

for j = 1:numOrderCands
  errMeans(j,:) = mean( results{j}, 1 );
  errStds(j,:) = std( results{j}, 1 );
end
stdErrs = errStds/sqrt(numExperiments);

figure;
orderStrings = cell(1,numel(selOrderIdxs));
for j = 1:numOrderCands
  if ismember(j, selOrderIdxs)
    plotFunc(nCands, errMeans(j,:), plotMarkers{j}, 'Color', plotColours{j}, ...
    'MarkerSize', MS, 'LineWidth', LW); 
    orderStrings{j} = sprintf('%d', orderCands(j));
    hold on,
  end
end
legend(orderStrings);

% Now plot error bars
if numExperiments > 1
  for j = 1:numOrderCands
    if ismember(j, selOrderIdxs)
      errorbar(nCands, errMeans(j,:), stdErrs(j,:), 'Color', plotColours{j});
    end
  end
end

minMean = min(min(errMeans));
maxMean = max(max(errMeans));
xlim( nCands(1)/2 + [0, nCands(end)]);
ylim( [0.9*minMean 1.1*maxMean] );

xlabel('Number of Data');
ylabel('Test Error');


set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');
