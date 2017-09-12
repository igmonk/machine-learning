function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 25 units in the hidden layer (2nd)
% 10 units (= num_labels) in the output layer (3rd)
% size(Theta1) = [25 401]
% size(Theta2) = [10 26]

X = [ones(m, 1) X]; % size(X) = [m 401]
z1 = X * Theta1'; %' size(z1) = [m 25]
a1 = sigmoid(z1); % size(a1) = [m 25]
a1 = [ones(m, 1) a1]; % size(a1) = [m 26]
z2 = a1 * Theta2'; %' size(z2) = [m 10];
a2 = sigmoid(z2);

% from help max:
%   -- max (X, [], DIM)
%   If the optional third argument DIM is present then operate along
%   this dimension.  In this case the second argument is ignored and
%   should be set to the empty matrix.
[p_max p_max_index] = max(a2, [], 2);
p = p_max_index;




% =========================================================================


end
