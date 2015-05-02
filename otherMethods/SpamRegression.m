function [pred] = Spam(Xtr, Ytr, lambda)
    max_iters = 1;
    [n, p] = size(Xtr);
    h = zeros(p,1);
    h = 0.1*ones(p,1);
    for j=1:p
        hx=median(abs(Xtr(:,j)-median(Xtr(:,j))))/0.6745*(4/3/n)^0.2;
        hy=median(abs(Ytr-median(Ytr)))/0.6745*(4/3/n)^0.2;
        %h(j)=sqrt(hy*hx);
    end
    alpha = mean(Ytr);
    kerf=@(z) exp(-z.*z/2)/sqrt(2*pi);
    fhat = zeros(n, p);
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

            %G = zeros(n,p);
            %for nn=1:n
            %    for k=setdiff(1:p,j)
            %        z=kerf( (X(nn,k)-X(:,k))/h(k) );
            %        G(nn,k) = sum(z.*Ytr)/sum(z);
            %    end
            %end
        end
        pred = alpha + sum(fhat,2);
        err = norm(Ytr - pred)^2
        figure; plot(Xtr, Pj);
    end
end

