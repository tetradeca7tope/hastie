function YPred = regTree(X, Y, Xte)
  tree = classregtree(X, Y);
  YPred = eval(tree, Xte);
end

