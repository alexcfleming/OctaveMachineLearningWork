function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies(m)  x num_features(n) matrix of movie features
%        Theta - num_users(u)  x num_features(n) matrix of user features
%        Y - num_movies(m) x num_users(u) matrix of user ratings of movies
%        R - num_movies(m) x num_users(u) matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
J = (1/2)*sum(sum((R.*((X*Theta')-Y)).^2));
X_grad = (R.*((X*Theta')-Y))*Theta;
Theta_grad = (R.*((X*Theta')-Y))'*X;
Jreg = (lambda/2)*sum(sum(Theta.^2))+(lambda/2)*sum(sum(X.^2));
J = J + Jreg;
Xgradreg = lambda.*X;
Thetagradreg = lambda.*Theta;
X_grad = X_grad + Xgradreg;
Theta_grad = Theta_grad + Thetagradreg;
% =============================================================
grad = [X_grad(:); Theta_grad(:)];

end
