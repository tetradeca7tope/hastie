function [optAlpha, optStats] = subGradAlpha(Ks, Y, lambda, params)

    % prelims
    n = size(Y, 1);
    M = size(Ks, 3);
    % Obtain optimisation params
    params = processOptParamsCommon(params, n, M);
    Alpha = zeros(n, M);
    KAlphas = zeros(n, M);

    % Set up book keeping
    prevObj = objectiveAlpha(Alpha, KAlphas, Y, lambda);
    prevStepSize = Inf;
    currBestObj = prevObj;
    optAlpha = Alpha;
    objHistory = prevObj;
    timeHistory = 0;
    startTime = cputime;

    for iter = 1:params.maxNumIters
        G1 = zeros(n, M);
        for g = 1:M
            %G1(:,g) = (1/n)*Ks(:,:,g)*(sum(KAlphas(:,setdiff(1:M,g)),2)-Y) ...
            %    + (1/(2*n))*Ks(:,:,g)*KAlphas(:,g);
            G1(:,g) = Ks(:,:,g)'*( ((sum(KAlphas(:,setdiff(1:M,g)),2)-Y)/n) + ...
                                   KAlphas(:,g)/(2*n) );
        end

        G2 = zeros(n, M);
        for g=1:M
            if norm(Alpha(:,g)) > 0
                G2(:,g) = KAlphas(:,g) / sqrt(Alpha(:,g)'*KAlphas(:,g));
            end
        end
        G2 = (lambda/M) * G2;
        G = G1 + G2;
        
        % perform update
        stepSize = min(10/(iter*(norm(G(:),2)+1e-2)), 2*prevStepSize);
        prevStepSize = stepSize;
        %stepSize = 1/ceil(iter/10);
        Alpha = Alpha - stepSize*G;
        for g=1:M
            KAlphas(:,g) = Ks(:,:,g) * Alpha(:,g);
        end

        % evaluate objective
        currObj = objectiveAlpha(Alpha, KAlphas, Y, lambda);
        currTime = cputime - startTime;
        if currObj < currBestObj
            currBestObj = currObj;
            optAlpha = Alpha;
        end
        objHistory = [objHistory; currBestObj];
        timeHistory = [timeHistory; currTime];

        % Print results out
        if params.optVerbose & mod(iter, params.optVerbosePerIter) == 0,
            fprintf(['SubGrad #%d (%0.3fs): currObj: %0.4e, ' ...
                    'bestObj: %.4e, stepSize: %e\n'], ...
                    iter, currTime, currObj, currBestObj, stepSize);
        end

        % Update
        prevObj = currObj;
        
        % Termination condition
        if abs( (currObj - prevObj) / currObj ) < params.tolerance
            %break;
        end

    end

    % statistics
    optStats.objective = objHistory;
    optStats.time = timeHistory;
end

function [obj] = objectiveAlpha(Alpha, KAlphas, Y, lambda)
    [n, M] = size(Alpha);
    pred_diff = Y - sum(KAlphas,2);
    g = (norm(pred_diff,2)^2) / (2*n);
    h = 0;
    for j = 1:M
        h = h + sqrt(Alpha(:,j)'*KAlphas(:,j));
    end
    h = (lambda/M) * h;
    obj = g + h;
    %fprintf('obj:%f g:%f h:%f \n', obj, g, h);
end
