function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% You need to return the following variables correctly.
errors = zeros(64,3);

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
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

Ctestset = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmatestset = [0.01 0.03 0.1 0.3 1 3 10 30];
k = 1; %declare row counter for errors matrix at 1

for i = 1:8;
  for j = 1:8;
    Ctemp = Ctestset(i);
    sigmatemp = sigmatestset(j);
    model = svmTrain(X, y, Ctemp, @(x1, x2) gaussianKernel(x1, x2, sigmatemp));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    errors(k,:) = [Ctemp sigmatemp error];
    k = k + 1;
    end
end

q = find(errors(:,3) == min(errors(:,3)));
C = errors(q,1);
sigma = errors(q,2);

% =========================================================================
end
