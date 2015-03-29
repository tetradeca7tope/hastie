function [obj] = fast_objective(alphas, y, K_alphas, lambda_1, lambda_2, lambda_3)
% Complexity O(nm)
    m = size(K_alphas, 2); 
    fit = y - sum(K_alphas, 2);
    obj_fit = 0.5*(fit'*fit);
    obj_1 = 0;
    for g=1:m
        alpha_g = alphas(:,g);
        obj_1 = obj_1 + 0.5*lambda_1*alpha_g'*K_alphas(:,g);
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
