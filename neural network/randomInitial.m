function [ output ] = randomInitial( size1, size2 )

epsilon_init = 0.12;

output = rand(size2 , size1+1)*2*epsilon_init- epsilon_init;

end

