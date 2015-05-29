function [Predtr, Predte] = SpamRegression(Xte, Xtr, Ytr, lambda)
    [n, p] = size(Xtr);
    max_iters = min(round(1000/p), 50);
    nte = size(Xte,1);
    h = zeros(p,1);
    for j=1:p
        hx=median(abs(Xtr(:,j)-median(Xtr(:,j))))/0.6745*(4/3/n)^0.2;
        if hx == 0, hx = 1; end
        hy=median(abs(Ytr-median(Ytr)))/0.6745*(4/3/n)^0.2;
        if hy == 0, hy = 1; end
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
        if norm(Predtr-oldPredtr,2)/norm(oldPredtr,2) < 1.0e-4
            break;
        end
        oldPredtr = Predtr;
    end
    fhatte = zeros(nte,p);
    for j=1:p
%         [~, unique_ix] = uniquetol(Xtr(:,j));
%         [~, unique_ix] = unique(Xtr(:,j));
%         fhatte(:,j) = spline(Xtr(:,unique_ix),fhat(:,unique_ix),Xte(:,j));
%         fhatte(:,j) = spline(Xtr(:,j), fhat(:,j), Xte(:,j));
%         predFunc = localPolyRegressionCV(Xtr(:,j), fhat(:,j));
%         fhatte(:,j) = predFunc(Xte(:,j));
        silvBw = 1.06 * std(Xtr(:,j)) * n^0.2;
        fhatte(:,j) = localPolyRegression(Xte(:,j), Xtr(:,j), fhat(:,j), ...
          silvBw, 0);
%         fhatte(:,j) = spline(Xtr(:,j), fhat(:,j), Xte(:,j));
    end
    Predte = alpha + sum(fhatte,2);
end

