

% Now plot the results out
minObjs = zeros(numMethods, 1);
maxObjs = zeros(numMethods, 1);
maxTimes = zeros(numMethods, 1);

figure;
for j = 1:numMethods
  numCurrIters = size(objHistories{j}, 1);
  plotFunc(timeHistories{j}, objHistories{j}, plotMarkers{j}, ...
    'Color', plotColours{j});
  hold on,

  % some book keeping
  minObjs(j) = objHistories{j}(end);
  maxObjs(j) = objHistories{j}(1);
  maxTimes(j) = timeHistories{j}(end);
end
minObjVal = min(minObjs);
maxObjVal = max(maxObjs);
maxTimeVal = max(maxTimes);
objDiff = maxObjVal - minObjVal;
xlim([0 maxTimeVal*1.05]);
ylim([minObjVal maxObjVal]);
legend(compareMethods);
title('Objective vs Time (seconds)');

set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');

figure;
for j = 1:numMethods
  numCurrIters = size(objHistories{j}, 1);
  plotFunc(1:numCurrIters, objHistories{j}, plotMarkers{j}, ...
    'Color', plotColours{j});
  hold on,

  % some book keeping
end
xlim([0 optParams.maxNumIters*1.1]);
ylim([minObjVal maxObjVal]);
legend(compareMethods);
title('Objective vs Iteration');

set(0,'defaultAxesFontName', 'Dejavu Sans')
  set(findall(gca, '-property', 'FontSize'), 'FontSize', 18, ...
    'fontWeight', 'bold');

