function [optBeta, optStats] = subGradMethod(Ls, Y, lambda, params)
% Implements subGradient method to optimise the objective.

  % prelims
  n = size(Y, 1);
  M = size(Ls, 3);
  % Create a function handle for the objective
  objective = @(arg) computeObjBeta(arg, Ls, Y, lambda);
  smoothObjGrad = @(arg) computeSmoothObjGradBeta(arg, Ls, Y);
  % Obtain optimisation params
  params = processOptParamsCommon(params, n, M);
  Beta = params.initBeta;

  % Set up book keeping
  prevObj = objective(Beta);
  currBestObj = prevObj;
  optBeta = Beta;
  objHistory = prevObj;
  timeHistory = 0;
  startTime = cputime;

  for iter = 1:params.maxNumIters

    % Determine a subgradient
    [~, G1] = smoothObjGrad(Beta); % smooth term
    G2 = l2NormSubGradAllBeta(Beta);
    G = G1 + (lambda/M) * G2;
    
    % perform update
    stepSize = 1/ceil(iter/10);
    Beta = Beta - stepSize * G;

    % evaluate objective
    currObj = objective(Beta);
    currTime = cputime - startTime;
    if currObj < currBestObj
      currBestObj = currObj;
      optBeta = Beta;
    end
    objHistory = [objHistory; currBestObj];
    timeHistory = [timeHistory; currTime];

    % Print results out
    if params.optVerbose & mod(iter, params.optVerbosePerIter) == 0,
      fprintf('SubGrad #%d (%0.3fs): currObj: %0.4f, bestObj: %.4f, stepSize: %e\n', ...
        iter, currTime, currObj, currBestObj, stepSize);
    end

    % Update
    prevObj = currObj;
    
    % Termination condition
    if abs( (currObj - prevObj) / currObj ) < params.tolerance
      break;
    end

  end

  % statistics
  optStats.objective = objHistory;
  optStats.time = timeHistory;

end


function G = l2NormSubGradAllBeta(Beta)
% Compute the subgradient of all Norms
  [n, M] = size(Beta);
  G = zeros(n, M);
  for i = 1:M
    G(:,i) = l2NormSubGrad( Beta(:,i) );
  end
end


function g = l2NormSubGrad(beta)
% Compute the subgradient of the l2 norm
  normBeta = norm(beta);
  if normBeta == 0
    g = zeros(size(beta,1), 1);
%     g = randn(size(beta,1), 1);
%     g = rand() * g / norm(g);
  else
    g = beta /normBeta;
  end
end

