% Plots results out

plotColours = {'b', 'g', 'r', 'k', 'c', 'y', 'm', [255 128 0]/255, ...
  [76, 0, 153]/253, [102 102 0]/255};
plotMarkers = {'o', '+', '*', 'x', 's', 'd', '^', 'p', '>', 'v'};
plotFunc = @semilogy;

errMeans = zeros(numDMCandidates, numNCands);
errStds =  zeros(numDMCandidates, numNCands);
legStrs = cell(numDMCandidates, 1);

for j = 1:numDMCandidates
  errMeans(j,:) = mean( results{j}, 1 );
  errStds(j,:) = std( results{j}, 1 );
  legStrs{j} = sprintf('(%d, %d)', dMVals(j, 1), dMVals(j,2));
end

figure;
for j = 1:numDMCandidates
  plotFunc(nCands, errMeans(j,:), plotMarkers{j}, 'Color', plotColours{j}); ...
  hold on,
end
legend(legStrs);

set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');

% Now plot error bars
if numExperiments > 1
  stdErrs = errStds/sqrt(numExperiments);
  for j = 1:numDMCandidates
    errorbar(nCands, errMeans(j,:), stdErrs(j,:), 'Color', plotColours{j});
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
