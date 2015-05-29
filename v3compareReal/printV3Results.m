% Print results

% fprintf('\n\n');

fprintf('\nResults:%s\n=========================================\n', dataset);
fprintf('nTr = %d, nTe = %d, D = %d\n\n', nTr, nTe, numDims);

msErrors = zeros(numRegAlgos, 1);
genStr1 = sprintf('%s ($%d$,$%d$)\n', dataset, numDims, nTr);
genStr2 = genStr1;
[~, sortedOrder] = sort(results);

for i = 1:numRegAlgos
  regAlgo = regAlgos{i}{1};
  fprintf('Method: %s, Err: %0.6f, Time:%0.2f s', regAlgo, results(i), times(i));

  if strcmp(regAlgo,'addKRR')
    fprintf(', Order: %d', addKrrOrder);
  end
  fprintf('\n');

  if i <= 11
    genStr1 = sprintf('%s& $%0.5f$ ', genStr1, results(i));
    if strcmp(regAlgo, 'addKRR')
      genStr1 = sprintf('%s(%d) ',genStr1, addKrrOrder);
    end
  else
    genStr2 = sprintf('%s& $%0.5f$ ', genStr2, results(i));
  end
end
genStr1 = sprintf('%s \\\\ \\hline', genStr1);
genStr2 = sprintf('%s \\\\ \\hline', genStr2);


fprintf('\n%s\n%s\n', genStr1, genStr2);

fprintf('\nSorted Order:%s\n=======================\n', dataset);
for j = 1:numRegAlgos
  i = sortedOrder(j);
  regAlgo = regAlgos{i}{1};
  fprintf('Method: %s, Err: %0.6e, Time:%0.2f s', regAlgo, results(i), times(i));

  if strcmp(regAlgo,'addKRR')
    fprintf(', Order: %d', addKrrOrder);
  end
  fprintf('\n');

end

