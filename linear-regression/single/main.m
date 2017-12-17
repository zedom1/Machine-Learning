%% Gradient Descent for single feature

%% step1 load data

data = load('data1.txt');
x = data(:,1);
y = data(:,2);
m = length(y); % the number of training examples

%% plot data1

figure;
subplot(1,3,1);
plot(x,y,'rx')
xlabel('Population of city');
ylabel('Profit');

%% Gradient Descent

% process data
x = [ones(m,1) x];
theta = zeros(2,1);

% Gradient Descent
iterations = 1500;
alpha = 0.01;
theta = gradientDescent(x,y,theta,alpha,iterations)

hold on
plot(x(:,2),x*theta,'--')
hold off

theta0 = linspace(-10,10,100);
theta1 = linspace(-1,4,100);

j_matrix = zeros(length(theta0),length(theta1));

for i = 1:length(theta0)
    for j = 1:length(theta1)
        t = [theta0(i);theta1(j)];
        j_matrix(i,j) = computeCost(x,y,t);
    end
end

j_matrix = j_matrix';
subplot(1,3,2);
surf(theta0,theta1,j_matrix)
xlabel('\theta0');
ylabel('\theta1');
zlabel('J(\theta)');

subplot(1,3,3);
contour(theta0,theta1,j_matrix,logspace(-2,3,20))
xlabel('\theta0');
ylabel('\theta1');
hold on;
plot(theta(1),theta(2),'rx','MarkerSize',10,'LineWidth',2)
