function [Predtr, Predte] = KnnRegression(Xte, Xtr, Ytr, k)
    [IDX,D] = knnsearch(Xtr,Xte,'K',k,'distance','seuclidean');
    ntr = size(Xtr,1);
    nte = size(Xte,1);
    InvD = 1./D;
    ScaledInvD = InvD ./ repmat(sum(InvD,2), [1 k]);
    Predte = sum((Ytr(IDX)) .* ScaledInvD, 2);

    Predtr = Xtr;
end
