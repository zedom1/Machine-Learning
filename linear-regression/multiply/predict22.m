function price = predict22(c,theta)

c = [ones(size(c,1),1) c];
price = c*theta;

end