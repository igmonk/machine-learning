function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% size(X) = [300 2]
% size(idx) = [300 1]

for ci = 1:K
  ci_current = (idx == ci); % size(ci_current) = [300 1]
  ci_matrix = repmat(ci_current, 1, n); % size(ci_matrix) = [300 2]
  X_ci = X .* ci_matrix; % size(X_ci) = [300 2]
  centroids(ci, :) = sum(X_ci) ./ sum(ci_current);
end




% =============================================================


end
