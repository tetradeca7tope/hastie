function [optAlpha, optStats] = subGradAlpha(Ks, Y, lambda, params)

    % prelims
    n = size(Y, 1);
    M = size(Ls, 3);
    % Create a function handle for the objective
    objective = @(arg) computeObjAlpha(arg, Ks, Y, lambda);

    % Obtain optimisation params
    params = processOptParamsCommon(params, n, M);
    smoothObjGrad = @(arg) computeSmoothObjGradAlpha(arg, Ks, Y);
    Alpha = zeros(n, M);

    % Set up book keeping
    prevObj = objective(Beta);
    currBestObj = prevObj;
    optBeta = Beta;
    objHistory = prevObj;
    timeHistory = 0;
    startTime = cputime;

    for iter = 1:params.maxNumIters
        [~, G1] = smoothObjGrad(Alpha);
        size(G1)
        G2 = zeros(n, M);
        for g=1:M
            if norm(Alpha(:,g)) > 0
                KAg = Ks(:,:,g)*Alpha(:,g)
                G2(:,g) = KAg / sqrt(Alpha(:,g)'*KAg);
            end
        end
        G2 = (lambda/M) * G2;
        G = G1 + G2;
        
        % perform update
        stepSize = 10/(iter*(norm(G(:),2)+1e-8));
        Alpha = Alpha - stepSize*G;

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
            fprintf(['SubGrad #%d (%0.3fs): currObj: %0.4e, ' ...
                    'bestObj: %.4e, stepSize: %e\n', ...
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
end

