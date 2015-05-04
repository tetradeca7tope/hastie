% Print results

% fprintf('\nResults:%s\n=========================================\n', dataset);
for i = 1:numRegAlgos
  regAlgo = regressionAlgorithms{i}{1};
  fprintf('Method: %s, Err: %0.6e\n', regAlgo, results(i) );
end
