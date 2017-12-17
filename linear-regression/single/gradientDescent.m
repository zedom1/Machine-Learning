function [theta,j] = gradientDescent(x,y,theta,alpha,iterations)

m = length(y);
j = zeros(iterations,1);

for i = 1:iterations
    tem1 = x'*(x*theta-y);
    temtheta = theta - (alpha/m)*tem1;
    theta = temtheta;
    j(i)=computeCost(x,y,theta);
end

end