function [obj] = objective(alphas, y, K_all, lambda_1, lambda_2, lambda_3)
    
    [~, ~, m] = size(K_all);
    fit = y;
    for k=1:m
        fit = fit - squeeze(K_all(:,:,k))*squeeze(alphas(:,k));
    end
    obj_fit = 0.5*(fit'*fit);
    obj = obj_fit;
    for k=1:m
        alpha_k = alphas(:,k);
        K_k = squeeze(K_all(:,:,k));
        obj = obj + 0.5*lambda_1*alpha_k'*K_k*alpha_k;
    end
    
    for k=1:m
        norm_2 = alphas(:,k)'*alphas(:,k);
        obj = obj + 0.5*lambda_2*norm_2 + lambda_3*sqrt(norm_2);
    end 
end