%{ 
	Author: Dhairya Dhondiyal
	
	Description:
	The program trains Neural net to recognize handwritten digits using backpropogation to Obtain the weights.
	The gradient descent is done using an open source library fmincg which minimizes theta efficiently.
	This is done since using "fminunc" or "for" cause time and memory issues. 
	
	Disclaimer:
    Copyright (C) - All Rights Reserved
    Unauthorized copying of this file, via any medium is strictly prohibited
    Proprietary and confidential
    Written by Dhairya Dhondiyal, November 2016
%}
clear;close all;clc;
%Force matlab warnings to stay hidden
do_braindead_shortcircuit_evaluation(0);
clc;
%Set Options
options=optimset('MaxIter',50);

%Cost function implements Back propogation and Regularization with lambda
function [J,grad]=costfunc(nn,X,y,lambda)
O1=reshape(nn(1:25*401),25,(401));
O2=reshape(nn((1+(25*401)):end),10,26);
J=0;
O1_grad=zeros(size(O1));
O2_grad=zeros(size(O2));
z2=[ones(size(X,1),1) X]*O1';
z3=[ones(size((1.0 ./(1.0+exp(-z2))),1),1) (1.0 ./(1.0+exp(-z2)))]*O2';

y2=eye(10)(y,:);
reg=(sum(sum(O1(:,2:end).^2))+sum(sum(O2(:,2:end).^2)))*(lambda/(2*size(X,1)));
J=1/size(X,1)*sum(sum(-1*y2.*log(1.0 ./(1.0+exp(-z3)))-(1-y2).*log(1-(1.0 ./(1.0+exp(-z3))))))+reg;

%Back propogation algorithm to find Weights/Thetas
for t=1:size(X,1)
z2=O1*[1;X(t,:)'];
z3=O2*[1;(1.0 ./(1.0+exp(-z2)))];
yy=([1:10]==y(t))';
%Partial Derivatives denoted by Di
D1=(1.0 ./(1.0+exp(-z3)))-yy;
D2=(O2'*D1).*[1; (1.0 ./(1.0+exp(-z2))).*(1-(1.0 ./(1.0+exp(-z2))))];
D2=D2(2:end);
O1_grad=O1_grad+D2*[1;X(t,:)']';
O2_grad=O2_grad+D1*[1;(1.0 ./(1.0+exp(-z2)))]';
end

%Gradients
O1_grad=(1/size(X,1))*O1_grad+(lambda/size(X,1))*[zeros(size(O1,1),1) O1(:,2:end)];
O2_grad=(1/size(X,1))*O2_grad+(lambda/size(X,1))*[zeros(size(O2,1),1) O2(:,2:end)];
grad=[O1_grad(:);O2_grad(:)];
end

%Load Data
X=load('training_x.txt');
y=load('training_y.txt');

%Randomized initial Weights
O1=rand(25,401)*2*0.12-0.12;
O2=rand(10,26)*2*0.12-0.12;
nn_init=[O1(:);O2(:)];

%Set Cost function parameters
GDwReg_params=@(p) costfunc(p,X,y,1);

%Uses fminunc open source minimizing function to implement gradient decent
[nn,cost]=fmincg(GDwReg_params,nn_init,options);

%Final weights after Minimization/Gradient descent
O1=reshape(nn(1:25*401),25,401);
O2=reshape(nn((1+(25*401)):end),10,26);

%Store predicted values in vector p
[temp,p]=max(1.0 ./(1.0+exp(-([ones(size(X,1),1) (1.0 ./(1.0+exp(-([ones(size(X,1),1) X]*O1'))))]*O2'))),[],2);

%Print Training Accuracy
fprintf('Training Accuracy: %f\n',mean(p==y)*100);

%Print Cost after max iteration.
fprintf('Cost: %f\n',cost(50));