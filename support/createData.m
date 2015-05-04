function createData()
    n_list = [100 200 300 400 500 600 700 800 900 1000];
    p = 100;
    num_n = length(n_list);
    num_reps = 20;
    for rep=1:num_reps
        for n_ix=1:num_n
            n = n_list(n_ix);
            [X, Y] = create(n,p);
            fname = sprintf('support-n%i-rep%i.mat', n, rep);
            save(fname, 'X', 'Y');
        end
    end
end


function [X, Y] = create(n, p)
    X = 4*rand(n, p) - 2;
    f1 = @(x) -2*sin(2*x);
    f2 = @(x) x.^2 - 1/3;
    f3 = @(x) x - 1/2;
    f4 = @(x) exp(-1*x) + exp(-1) - 1;
    g1 = @(x1,x2) f1(x1.*x2);
    g2 = @(x1,x2) f2(x1.*x2);
    g3 = @(x1,x2) f3(x1.*x2);
    g4 = @(x1,x2) f4(x1.*x2);
    Y = f1(X(:,1)) + f2(X(:,2)) + f3(X(:,3)) + f4(X(:,4)) + ...
        g1(X(:,5), X(:,6)) + g2(X(:,7),X(:,8)) + ...
        g3(X(:,9),X(:,10)) + g4(X(:,11),X(:,12)) + randn(n,1);
    
end


