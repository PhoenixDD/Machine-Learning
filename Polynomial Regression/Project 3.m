%{ 
	Author: Dhairya Dhondiyal
	
	Description:
	Description is there in the word file.
	
	Disclaimer:
    Copyright (C) - All Rights Reserved
    Unauthorized copying of this file, via any medium is strictly prohibited
    Proprietary and confidential
    Written by Dhairya Dhondiyal, October 2016
%}
clear;clear all;clc;

%Load cv_x and cv_y
cv_x=load('cv_x.txt');
cv_y=load('cv_y.txt');
cv_x=[ones(size(cv_x)) cv_x];

%Take rows 2-12 and calculate the errors after predection using ordianry least squares linear regression.
for i=2:12
training_x=load('training_x.txt');
training_y=load('training_y.txt');
training_x=training_x(1:i,:);
training_y=training_y(1:i,:);
training_x=[ones(size(training_x)) training_x];
theta=inv(training_x'*training_x)*training_x'*training_y;
training_error(i-1)=sum((training_x*theta-training_y).^2)/(2*length(training_y));
cv_error(i-1)=sum((cv_x*theta-cv_y).^2)/(2*length(cv_y));
end

%Plot the Learning curve for degree 1 polynomial.
plot(1:length(training_y)-1,training_error,1:length(training_y)-1,cv_error);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 12 0 120]);

%Convert the training_x to polynomial of degree 6.
training_x_poly=zeros(size(training_x,1),6);
training_x_poly(:,1)=training_x(:,2);
for i=2:6
training_x_poly(:,i)=training_x(:,2).*training_x_poly(:,i-1);
end
mean_training_x_poly=mean(training_x_poly);
training_x_poly=bsxfun(@minus,training_x_poly,mean_training_x_poly);
standard_deviation=std(training_x_poly);
training_x_poly=bsxfun(@rdivide,training_x_poly,standard_deviation);

%Convert cv_x to polynomial of degree 6.
cv_x_poly=zeros(size(cv_x,1),6);
cv_x_poly(:,1)=cv_x(:,2);
for i=2:6
cv_x_poly(:,i)=cv_x(:,2).*cv_x_poly(:,i-1);
end
cv_x_poly=bsxfun(@minus,cv_x_poly,mean_training_x_poly);
cv_x_poly=bsxfun(@rdivide,cv_x_poly,standard_deviation);
cv_x_poly=[ones(size(cv_x_poly,1),1) cv_x_poly];

%Take rows 2-12 and calculate the errors after predection using ordianry least squares polynomial linear regression.
for i=2:12
training_x=training_x_poly(1:i,:);
training_y=load('training_y.txt');
training_y=training_y(1:i,:);
training_x=[ones(size(training_x,1),1) training_x];
theta=pinv(training_x)*training_y;
training_error(i-1)=sum((training_x*theta-training_y).^2)/(2*length(training_y));
cv_error(i-1)=sum((cv_x_poly*theta-cv_y).^2)/(2*length(cv_y));
end

%Plot the Learning curve for degree 6 polynomial.
figure(2);
plot(1:length(training_y)-1,training_error,1:length(training_y)-1,cv_error);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 12 0 170]);
