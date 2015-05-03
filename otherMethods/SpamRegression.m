function [Predtr, Predte] = Spam(Xte, Xtr, Ytr, lambda)
    max_iters = 100;
    [n, p] = size(Xtr);
    nte = size(Xte,1);
    h = zeros(p,1);
    for j=1:p
        hx=median(abs(Xtr(:,j)-median(Xtr(:,j))))/0.6745*(4/3/n)^0.2;
        hy=median(abs(Ytr-median(Ytr)))/0.6745*(4/3/n)^0.2;
        h(j)=sqrt(hy*hx);
    end
    alpha = mean(Ytr);
    kerf=@(z) exp(-z.*z/2)/sqrt(2*pi);
    fhat = zeros(n, p);
    oldPredtr = alpha*ones(n,1);
    for iter=1:max_iters
        for j=1:p
            Rj = Ytr - alpha - sum(fhat(:,setdiff(1:p,j)),2);
            Pj = zeros(n,1);
            Xj = Xtr(:,j);
            for nn=1:n
                z = kerf( (Xj(nn)-Xj)/h(j) );
                Pj(nn) = sum(z.*Rj)/sum(z);
            end
            sj = sum(Pj.^2)/n;
            fhat(:,j) = max(0, 1 - lambda/sj)*Pj;
            fhat(:,j) = fhat(:,j) - mean(fhat(:,j));
        end
        Predtr = alpha + sum(fhat,2);
        if norm(Predtr-oldPredtr,2)/norm(oldPredtr,2) < 0.001
            break;
        end
        oldPredtr = Predtr;
    end
    fhatte = zeros(nte,p);
    for j=1:p
        fhatte(:,j) = spline(Xtr(:,j),fhat(:,j),Xte(:,j));
    end
    Predte = alpha + sum(fhatte,2);
    %Predte = spline(Xtr,Predtr,Xte);
end

