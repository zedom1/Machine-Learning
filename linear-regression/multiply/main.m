%% Gradient Descent with multiply features:
% step1 load data

data = load('data2.txt');
x = data(:,1:size(data,2)-1);
y = data(:,size(data,2));
m = length(y);

fprintf('The first ten examples\n')
fprintf('x = [%.1f %.1f], y =%.1f \n',[x(1:10,:) y(1:10,:)]')

%% Gradient Descent

% process data
[x mu sigma] = featureNormalize(x);
x = [ones(m,1) x];

% Gradient Descent
alpha = 0.01;
iterations = 1000;
theta = zeros(3,1);
[theta,j] = gradientDescentMulti(x,y,theta,alpha,iterations);

figure;
plot(1:numel(j),j,'-b','LineWidth',2);
xlabel('Number of iterations');
ylabel('Cost J');

tester = [1650,3];
price = predict11(tester,mu,sigma,theta);
fprintf('With the specific input [%f %f],\nwe predict the price is %f\n',[tester price]')


%% Normal Equations

data = load('data2.txt');
x = data(:,1:size(data,2)-1);
y = data(:,size(data,2));
m = length(y);
x = [ones(m,1) x];
theta = normalEqn(x,y);

c = [1650,3];
price = predict22(c,theta);
fprintf('With the specific input [%f %f],\nwe predict the price is %f\n',[c price]')

