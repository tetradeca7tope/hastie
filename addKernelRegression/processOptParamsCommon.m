function params = processOptParamsCommon(params, n, M)

  if isempty(params), params = struct();
  end

  if ~isfield(params, 'initBeta'),
    params.initBeta = zeros(n, M);
    fprintf('Initialising with all Zeros.\n');
  end
  if ~isfield(params, 'useAcceleration'),
    params.useAcceleration = true;
    fprintf('Using Acceleration.\n');
  end
  if ~isfield(params, 'stepSizeCriterion'), 
    params.stepSizeCriterion = 'backTracking';
    fprintf('Using Backtracking Line Search\n');
  end

  if ~isfield(params, 'tolerance'), params.tolerance = 1e-8;
  end
  if ~isfield(params, 'maxNumIters'), params.maxNumIters = 1e5;
  end
  if ~isfield(params, 'verbose'), params.verbose = true;
  end
  if ~isfield(params, 'verbosePerIter'), params.verbosePerIter = 25;
  end
  if ~isfield(params, 'optVerbose'), params.optVerbose = false;
  end
  if ~isfield(params, 'optVerbosePerIter'), params.optVerbosePerIter = 25;
  end

end

