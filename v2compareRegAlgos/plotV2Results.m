% Plots results out

plotColours = {'r', 'g', 'b', 'k', 'c'};
plotMarkers = {'-x', '-o', '-*', '-s', '-^'};
plotFunc = @semilogy;

errMeans = zeros(numRegAlgos, numCandidates);
errStds = zeros(numRegAlgos, numCandidates);

for j = 1:numRegAlgos
  errMeans(j,:) = mean( results{j}, 1 );
  errStds(j,:) = std( results{j}, 1 );
end

figure;
for j = 1:numRegAlgos
  plotFunc(nCands, errMeans(j,:), plotMarkers{j}, 'Color', plotColours{j}); ...
  hold on,
end
legend(regressionAlgorithms);

set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');

% Now plot error bars
if numExperiments > 1
  stdErrs = errStds/sqrt(numExperiments);
  for j = 1:numRegAlgos
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
