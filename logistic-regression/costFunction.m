function [ J , grad ] = costFunction( theta, X, y, lambda )

% Compute cost and gradient for logistic regression regularization

% the number of training examples;
m = length(y);

% normal logistic cost function
J = 1.0/(m*1.0)*((-(y'*log(sigmoid(X*theta)))-(1-y)'*log(1-sigmoid(X*theta)))*1.0);

% regularization (normally ignore theta(1):the parameter for feature==1)
J = J+ lambda*1.0/(2.0*m)*(theta'*theta*1.0-theta(1)*theta(1));

% normal logistic gradient calculation
grad= 1.0/m*(X'*(sigmoid(X*theta)-y));

% regularization (normally ignore theta(1):the parameter for feature==1)
mm = length(theta);
grad(2:mm) = grad(2:mm) + lambda/m*theta(2:mm);

end

