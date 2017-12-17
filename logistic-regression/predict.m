function p = predict(theta, X)

p = sigmoid(X*theta);

for i = [1:length(p)]
    if p(i)>=0.5
        p(i)=1;
    else
        p(i)=0;
    end
end

end