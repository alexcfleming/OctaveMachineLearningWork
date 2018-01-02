function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
theta = pinv(X'*X)*(X'*y);
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
% ============================================================

end
