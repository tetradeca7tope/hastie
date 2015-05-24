% Plots results out

plotColours = {'b', 'g', 'r', 'k', 'c', 'y', 'm', [255 128 0]/255, ...
  [76, 0, 153]/253, [102 102 0]/255};
% plotMarkers = {'o', '+', '*', 'x', 's', 'd', '^', 'p', '>', 'v'};
plotMarkers = {'o-', '+-', '*-', 'x-', 's-', 'd-', '^-', 'p-', '>-', 'v-'};
plotFunc = @semilogy;

MS = 6;
LW = 2;

errMeans = zeros(numRegAlgos, numNCandidates);
errStds = zeros(numRegAlgos, numNCandidates);

for j = 1:numRegAlgos
  errMeans(j,:) = mean( results{j}, 1 );
  errStds(j,:) = std( results{j}, 1 );
end

figure;
for j = 1:numRegAlgos
  plotFunc(nCands, errMeans(j,:), plotMarkers{j}, 'Color', plotColours{j}, ...
  'MarkerSize', MS, 'LineWidth', LW); 
  hold on,
end
legend(regressionAlgorithms);

% Now plot error bars
if numExperiments > 1
  stdErrs = errStds/sqrt(numExperiments);
  for j = 1:numRegAlgos
    errorbar(nCands, errMeans(j,:), stdErrs(j,:), 'Color', plotColours{j});
  end
end

minMean = min(min(errMeans));
maxMean = max(max(errMeans));
xlim( 0.9 * nCands(1) + [0, nCands(end)]);
ylim( [0.9*minMean 1.1*maxMean] );

xlabel('Number of Data');
ylabel('Test Error');


set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');
