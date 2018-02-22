function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

x1 = X(:,1)';
x2 = X(:,2)';

Min_error = 10000;
New_C = C;
New_Sigma = sigma;

%C_Trials = 0:5;
%Sigma_Trials = sigma*(1:10);

for i = 0:5
  c = C*(10^i);
  
  for j = 1:10
    error = 10000;
    
    s = sigma*j;
    
    printf("\n*** Training for C = %d and sigma = %f ***\n",c,s);
    model = svmTrain(X, y, c, @(x1,x2) gaussianKernel(x1,x2, s));
    
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));

    printf("Completed training for C = %d and sigma = %f. error = %f,Min error = %f.\n",c,s,error,Min_error);
    
    if (error < Min_error) #|| ((error == Min_error) && (c > New_C || s > New_Sigma))
      New_C = c;
      New_Sigma = s;
      Min_error = error;
      
      printf("Found better fit: Error = %f, new C = %d, new Sigma = %f\n",Min_error,New_C,New_Sigma);
    end; 
  end;
  
  C= New_C;
  Sigma = New_Sigma;
end;


% =========================================================================

end
