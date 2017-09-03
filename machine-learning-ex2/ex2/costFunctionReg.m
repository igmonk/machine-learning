function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
term1 = (-y) .* log(h);
term2 = (1 - y) .* log(1 - h);
thetaReg = theta(2:end); % use all except theta(0)
regTerm = (lambda / (2 * m)) * sum(thetaReg .^ 2);
J = sum(term1 - term2) / m + regTerm;

gradRegTerm = (lambda / m) * theta;
gradRegTerm(1) = 0; % regularization term for theta(0) should be equal to zero
grad = (1 / m) * (X' * (h - y)) + gradRegTerm;

% =============================================================

end
