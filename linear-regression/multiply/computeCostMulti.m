function j = computeCostMulti(x,y,theta)

m = length(y);
mm = x*theta-y;
j = 1/(2*m)*(mm'*mm);

end