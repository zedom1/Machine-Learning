function j = computeCost(x,y,theta)

m = length(y);
tem1 = x*theta-y;
tem1 = tem1.^2;
j = 1/(2*m)*sum(tem1);

end