function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
% =========================================================================
m = length(y); % number of training examples
h = X*theta;
J = (1/(2*m))*(sum(((h - y).^2)));
grad = (1/m)*(X'*(h - y));
theta(1) = 0; % set the bias terms to zero
Jreg = (lambda/(2*m))*(sum(theta.^2));
gradreg = (lambda/m)*theta;
% add regularization terms to J and grad
J = J + Jreg;
grad = grad + gradreg;
% =========================================================================
grad = grad(:);

end
