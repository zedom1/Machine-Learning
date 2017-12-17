function theta = normalEqn(x,y)

theta = pinv(x'*x)*x'*y;
end