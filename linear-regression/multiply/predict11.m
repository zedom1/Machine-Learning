function price = predict11(c,mu,sigma,theta)

c = (c-mu)./sigma;
c = [ones(1,1) c];
size(c)
price = c*theta;

end