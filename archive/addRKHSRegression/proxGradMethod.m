function [currBestAlpha, stats] =  ...
  addKernelRidgeProxGradDesc(Ks, Y, lambda1, lambda2, params)
% Implements Proximal Gradient Descent for Additive RKHS regression with
% acceleration and backtracking.
% The objective is 
% 0.5||Y - \sum_j Kj*alpha_j||_2^2 + 0.5*lambda1\sum_j alphaj'*Kj*alphaj
%   + lambda2 \sum_j ||alpha_j||_2
% The third term is a group sparsity penalty.
% Ks: The Kernel matrices for each Kernel
% Y: The labels
% lambda1, lambda2: The coeffs for the RKHS and group sparsity penalties.
% Params may contain the following
%   - initAlpha: Initialization for Alpha (Default O)
%   - tolerance: termination criterion (Default 1e-6)
%   - maxNumIters: A Maximum Number of Iterations (Default: 1e9)
%   - verbose: Whether to print progress (Default: true)

  % Optimisation parameters
  beta = 0.9; % damping factor for backtracking line search

  % prelims
  n = size(Y, 1);
  M = size(Ks, 3);
  % Create a function handle for the objective
  objective = @(arg) computeObj(arg, Ks, Y, lambda1, lambda2);
  smoothObjGrad = @(arg) computeSmoothObjGrad(arg, Ks, Y, lambda1);
  % Check for optimisation parameters
  params = processOptParamsCommon(params, n, M);

  % Params for accelerated proximal gradient method
  terminateNow = false;
  Alpha2 = params.initAlpha; % 2 steps before
  Alpha1 = params.initAlpha; % 1 step before
  iter = 0;
  stepSize = 1;
  prevObj = objective(Alpha1);
  currBestObj = inf;
  stepSize = 0.1;
  objHistory = prevObj;
  timeHistory = 0;
  startTime = cputime;

  % Perform Descent steps
  while (iter < params.maxNumIters) && (~terminateNow)

    iter = iter + 1;
    if params.useAcceleration
      V = Alpha1 + (iter - 2)/(iter + 1) * (Alpha1 - Alpha2); % Acceleration
    else
      V = Alpha1; % Just gradient Descent
    end
    [~, G] = smoothObjGrad(V);

    if strcmp(params.stepSizeCriterion, 'backTracking')
      [Alpha, stepSize] = ...
        backTracking(V, G, smoothObjGrad, stepSize, lambda2, beta);
    elseif strcmp(params.stepSizeCriterion, 'fixedStepSize')
      Alpha = fixedStepSize(V, G, params.stepSize, lambda2);
    else
      error('Unknown Step Size Criterion');
    end

    % Compute objective,
    currObj = computeObj(Alpha, Ks, Y, lambda1, lambda2);
    currTime = cputime - startTime;
    if currObj < currBestObj,
      currBestObj = currObj;
      currBestAlpha = Alpha;
    end
    objHistory = [objHistory; currBestObj];
    timeHistory = [timeHistory; currTime];

    % Termination condition
    if abs( (currObj - prevObj)/ currObj ) < params.tolerance,
      terminateNow = true;
    end

    % Print Results out
    if params.verbose & mod(iter, params.verbosePerIter) == 0,
    fprintf('#%d (%0.4f): currObj: %.4f, currBestObj: %0.4f, stepSize: %e\n', ...
      iter, currTime, currObj, currBestObj, stepSize);
    end

    % Update Alpha1 and Alpha2
    Alpha2 = Alpha1;
    Alpha1 = Alpha;
    prevObj = currObj;

%     debugIdxs = [3 4 8 12];
%     size(Alpha), Alpha(debugIdxs, :), G(debugIdxs, :), pause,

  end

  stats.objective = objHistory;
  stats.time = timeHistory;

end


function [Alpha, stepSize] = ...
  backTracking(V, G, smoothObjGrad, stepSize, lambda2, beta)
% smObj is the smooth part of the objective
  gV = smoothObjGrad(V);
  Alpha = groupSparsityProxOp(V - stepSize * G, stepSize * lambda2);
  while smoothObjGrad(Alpha) > gV + sum(sum( G .* (Alpha-V) )) + ...
    0.5/stepSize * norm(Alpha-V, 'fro') 
    stepSize = beta * stepSize;
    Alpha = groupSparsityProxOp(V - stepSize * G, stepSize * lambda2);
  end
end


function [Alpha] = fixedStepSize(V, G, stepSize, lambda2)
  Alpha = groupSparsityProxOp(V - stepSize * G, stepSize * lambda2);
end

