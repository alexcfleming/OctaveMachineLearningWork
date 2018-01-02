function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables and y matrix
m = size(X, 1);
y_matrix = eye(num_labels)(y,:); %produces m x num_labels(r) size matrix
         
% forward propogation
a1 = [ones(m, 1) X];
z2 = a1*Theta1'; % dimensions m x h
a2 = [ones(m, 1) (sigmoid(z2))]; % m x (h+1)
z3 = a2*Theta2';
a3 = sigmoid(z3); % dimensions m x num_labels (r)

% use y_matrix which is m x num_labels and a3 which is h, to produce cost
J = (1/m)*sum(sum((log(a3).*(-y_matrix))-(log(1-a3)).*(1-y_matrix)));
Theta1reg = Theta1(:, 2:end); % strips first columns from thetas
Theta2reg = Theta2(:, 2:end); % takes away a columns to produce r x h dimensions
Jreg = (lambda/(2*m))*(sum(sum(Theta1reg.^2))+sum(sum(Theta2reg.^2)));
J = J + Jreg; % adds cost and regularization term to cost

%backpropagation
d3 = a3 - y_matrix; % m x r - m x r = m x r
d2 = (d3*Theta2reg).*sigmoidGradient(z2); % (m x r * r x h) .* m x h = m x h
Delta1 = d2'*a1; % (m x h)T * m x n = h x n
Delta2 = d3'*a2; % (m x r)T * m x (h+1) = r x h+1

%gradients (unregularized)
Theta1_grad = (1/m)*Delta1; % h x n
Theta2_grad = (1/m)*Delta2; % r x (h+1)

%regularization of gradient terms
Theta1(:,1) = 0; % h x n
Theta2(:,1) = 0; % r x (h + 1)
Theta1scale = (lambda/m)*Theta1;
Theta2scale = (lambda/m)*Theta2;

%update gradient terms
Theta1_grad = Theta1_grad + Theta1scale;
Theta2_grad = Theta2_grad + Theta2scale;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
