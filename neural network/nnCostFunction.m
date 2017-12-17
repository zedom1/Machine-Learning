function [ J ,grad ] = nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y ,lambda )

% costfuncition and gradient with regularization

% reshape theta1 and theta2

Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, input_layer_size+1);

Theta2 = reshape(nn_params(hidden_layer_size*(input_layer_size+1)+1:end), num_labels, hidden_layer_size+1);

m = size(X,1);

%%  Feedforward

% add ones to the first raw of X
X = X';
X = [ones(1,size(X,2));X];
a1 = X;

z2 = Theta1*a1;
a2 = [ones(1,size(z2,2));sigmoid(z2)];

z3 = Theta2*a2;
a3 = sigmoid(z3);

yy = double(y==1);
for i =[2:num_labels]
    yy = [yy, double(y==i)];
end

delta3 = a3-yy';
delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);

Theta1_grad = 1/m * delta2*a1';
Theta1_grad(:, 2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);

Theta2_grad = 1/m * delta3*a2';
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

J = 1/m*sum(sum((-yy'.*log(a3)-(1-yy').*log(1-a3))));

%% Regularization with the cost function

Theta1_bias = sum(Theta1(:,1).^2);
Theta2_bias = sum(Theta2(:,1).^2);

Theta1_sum = sum(sum(Theta1.^2))-Theta1_bias;
Theta2_sum = sum(sum(Theta2.^2))-Theta2_bias;

J = J + lambda/(2*m)*(Theta1_sum+Theta2_sum);

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];

end

