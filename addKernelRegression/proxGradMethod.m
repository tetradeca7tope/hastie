function [optBeta, optStats] = proxGradMethod(Ls, Y, lambda, params)
% Implements proximal Gradient Method to optimise the objective

% Ls: The cholesky decomposition of each kernel matrix
% Y: The labels:
% lambda: The coefficients for the regularization penalty.

  % Optimisation Parameters
  dampFactor = 0.9; % for backtracking line search

  % prelims
  n = size(Y, 1);
  M = size(Ls, 3);
  % Create function handle for the objective
  objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
  smoothObjGrad = @(arg) computeSmoothObjGradBeta(arg, Ls, Y);
  params = processOptParamsCommon(params, n, M);

  useBackTracking = strcmp(params.stepSizeCriterion, 'backTracking');

  % Params for (Accelerated) Prox gradient method
  Beta2 = params.initBeta;
  Beta1 = params.initBeta;
  stepSize = 1;
  prevObj = objective(Beta1);
  currBestObj = prevObj;
  objHistory = prevObj;
  timeHistory = 0;
  startTime = cputime;
  
  % Perform Descent steps
  for iter = 1:params.maxNumIters
    
    if params.useAcceleration
      V = Beta1 + (iter - 2)/(iter + 1) * (Beta1 - Beta2);
    else
      V = Beta1;
    end
    [smObj, G] = smoothObjGrad(V);

    % Now determine step size
    if useBackTracking
      [Beta, stepSize] = ...
        backTracking(V, smObj, G, smoothObjGrad, stepSize, lambda, dampFactor);
    else
      % Use the stepSize given in params
      Beta = fixedStepSize(V, G, params.stepSize, lambda);
    end

    % Compute objective
    currObj = objective(Beta);
    currTime = cputime - startTime;
    if currObj < currBestObj
      currBestObj = currObj;
      currBestBeta = Beta;
    end
    objHistory = [objHistory; currBestObj];
    timeHistory = [timeHistory; currTime];

    % Print results
    if params.optVerbose & mod(iter, params.optVerbosePerIter) == 0,
      fprintf('ProxG #%d (%0.3f): currObj: %0.4e, currBestObj: %0.4e, stepSize: %e\n', ...
        iter, currTime, currObj, currBestObj, stepSize);
    end

    % Termination condition
    if abs( (currObj - prevObj)/ currObj ) < params.tolerance,
      fprintf('Terminating Prox Gradient Descent in %d iterations.\n', ...
        iter);
      break;
    end

    % Update Beta1 and Beta2
    Beta2 = Beta1;
    Beta1 = Beta;
    prevObj = currObj;

  end

  optStats.objective = objHistory;
  optStats.time = timeHistory;
  optBeta = Beta;

end


% Back tracking
function [Beta, stepSize] = ...
  backTracking(V, smObj, G, smoothObjGrad, stepSize, lambda, dampFactor)
% smObj is the smooth part of the objective
  Beta = groupSparsityProxOp(V - stepSize * G, stepSize * lambda);
  while smoothObjGrad(Beta) > smObj + sum(sum( G .* (Beta-V) )) + ...
    0.5/stepSize * norm(Beta-V, 'fro')
    stepSize = dampFactor * stepSize;
    Beta = groupSparsityProxOp(V - stepSize * G, stepSize * lambda);
  end
end


% Fixed Step Size
function [Beta] = fixedStepSize(V, G, stepSize, lambda)
  Beta = groupSparsityProxOp(V - stepSize * G, stepSize * lambda);
end

