function predFunc = nlls(X, Y)
  model = fitnlm(X, Y);
  predFunc = @(arg) predict(model, arg); 
end

