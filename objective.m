function [obj] = objective(alphas, y, K_all, lambda_1, lambda_2, lambda_3)
% Complexity O(n^2 m)
    
    [~, ~, m] = size(K_all);
    fit = y;
    for g=1:m
        fit = fit - K_all(:,:,g)*alphas(:,g);
    end
    obj_fit = 0.5*(fit'*fit);
    obj_1 = 0;
    for g=1:m
        alpha_g = alphas(:,g);
        obj_1 = obj_1 + 0.5*lambda_1*alpha_g'*K_all(:,:,g)*alpha_g;
    end

    obj_2 = 0;
    for g=1:m
        norm_2 = alphas(:,g)'*alphas(:,g);
        obj_2 = obj_2 + 0.5*lambda_2*norm_2;
    end

    obj_3 = 0;
    for g=1:m
        obj_3 = obj_3 + lambda_3*norm(alphas(:,g),2);
    end 
    obj = obj_fit + obj_1 + obj_2 + obj_3;
    %fprintf('fit=%f 1=%f 2=%f 3=%f sum=%f \n', ...
    %    obj_fit, obj_1, obj_2, obj_3, obj);
end
