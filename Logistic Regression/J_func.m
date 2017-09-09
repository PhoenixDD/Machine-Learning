%Cost function to find new g/gradient and J/cost
function[J,g]=J_func(O,X,y)
%Gradient descent algorithm.
z=X*O;
H=1./(1+e.^-z);
J=(1/length(y))*(-y'*log(H)-(1-y)'*log(1-H));
g=(1/length(y))*X'*(H-y);
end