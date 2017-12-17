function [ output ] = sigmoidGradient( input )

sig = sigmoid(input);
output = sig.*(1-sig);

end

