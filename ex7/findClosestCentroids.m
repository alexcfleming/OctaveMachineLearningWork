function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K and m
K = size(centroids, 1);
m = size(X, 1);

% create index 
idx = zeros(size(X,1), 1);
distance = zeros(m,K);
for i = 1:K;
  diffs = bsxfun(@minus, X, centroids(i,:)); %apply X as a whole to each centroid
  distance(:,i) = sum(diffs.^2,2); %adds the extra ,2 in sum to sum across columns
  endfor
[dumb idx] = min(distance, [], 2); 

end

