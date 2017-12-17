%% Neural Network

%% Setup paramters

input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

%% Load data

load('data.mat');
m = size(X,1);

% Displace 100 random data points
selected = randperm(size(X,1));
selected = selected(1:100);

displayData(X(selected, :));

%% Initial parameters

Theta1 = randomInitial(input_layer_size, hidden_layer_size);
Theta2 = randomInitial(hidden_layer_size, num_labels);
nn_params = [Theta1(:); Theta2(:)];

%% Train network

options = optimset('MaxIter', 50);

lambda = 1;

costFunction = @(p)nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X,y, lambda);

[nn_params , cost ] = fmincg(costFunction, nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size,(input_layer_size+1));
Theta2 = reshape(nn_params(hidden_layer_size*(input_layer_size+1)+1:end), num_labels, hidden_layer_size+1);


%% Prediction

pre = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pre == y)) * 100);


