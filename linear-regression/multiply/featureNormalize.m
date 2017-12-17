function [x mu sigma]= featureNormalize(x)

mu = mean(x,1);
sigma = std(x,0,1);

mu_matrix = mu(ones(length(x),1),:);
sigma_matrix = sigma(ones(length(x),1),:);

x = x-mu_matrix;
x = x./sigma_matrix;

end