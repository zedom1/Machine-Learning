%% Logistic regression with regularization

%% Load data

data = load('data.txt');
X = data(:, [1,2]);
y = data(:, 3);

figure;
subplot(1,2,1);
plotData(X,y);
% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%% Initial data

% Add polynomial features
% Also adds a column of ones
X = mapFeature(X(:,1), X(:,2));

% fitting parameters
theta = zeros(size(X,2),1);

% regularization parameter lambda
lambda = 1;

[cost , grad] = costFunction(theta,X, y, lambda);

fprintf('Cost at initial theta(zeros):%f\n' , cost);
fprintf('Gradient at initial theta(zeros):(first five)%f\n',grad(1:5));

%% Regularization and accuracies

options = optimset('GradObj','on','MaxIter',400);

[theta, J, exit_flag] = fminunc(@(t)(costFunction(t,X,y, lambda)), theta,options );
subplot(1,2,2);

hold on;
plotDecisionBoundary(theta, X,y);
title(sprintf('lambda=%g',lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



