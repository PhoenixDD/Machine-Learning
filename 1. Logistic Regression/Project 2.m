%{ 
	Author: Dhairya Dhondiyal
	
	Description:
	This program uses gradient descent to solve logistic regression in octave.
	It uses the built-in fminunc function to minimize the cost function.
	Uses 'trainingdata.txt' to train the model for bank notes identification.
	The cross validation is done using random rows of 'training_data' which have 90% data randomly picked from the 'training_data.txt' to train and other 10% data to check.
	The cross-validation result is the average of errors in all the 10 iterations.
	
	Disclaimer:
    Copyright (C) - All Rights Reserved
    Unauthorized copying of this file, via any medium is strictly prohibited
    Proprietary and confidential
    Written by Dhairya Dhondiyal, September 2016
%}
clear all;
clear;
clc;

average_error=0;

%Training data files with commas instead of spaces for seperation.
data=load('training_data.txt');
%Includes loops for original accuracy and 10 cross-validation loops.
for i=1:11
%Load data into the matrix without label/Class, Load class/Label in a column vector.
X=data(:,[1, 2,3,4]);y=data(:,5);
%Randomize rows for loop 2-11. Insert 90% data, that is 1234 lines
if i!=1
randomized_data=data(randperm(size(data,1)),:);
X=randomized_data(1:1234,[1, 2,3,4]);y=randomized_data(1:1234,5);
end
%Load 1's in the first column instead of labels.
X=[ones(size(X,1),1) X];
%initialize O/theta with zeroes for number of columns of X.
O=zeros(size(X,2),1);
%initialize gradient with zeroes for number of columns of X.
grad=zeros(size(O));

%in-built options for minimizing the cost function.
options=optimset('GradObj','on','MaxIter',400);
%in-built function for minimizing the cost function.
[O,cost]=fminunc(@(t)(J_func(t,X,y)),O,options);
%Randomize rows for loop 2-11. Insert 10% data, that is line 1235 to line 1372 line.
if i!=1
X=randomized_data(1235:1372,[1, 2,3,4]);y=randomized_data(1235:1372,5);
X=[ones(size(X,1),1) X];
end
p=zeros(size(X,1),1);

%class/label identification for each new probablity of X.
z=X*O;
p=1./(1+e.^-z)>=0.5;

if i==1
%Equate new labels with original labels to find out the accuracy and subtract with 100 to get error rate and print.
fprintf('Actual Error(using entire data for training and testing): %f\n\n',100-mean(p==y)*100);
else
%Equate new labels with original labels to find out the accuracy and subtract with 100 to get error rate for each cross-validation iteration and print.
fprintf('Error on #%dth iteration of cross validation: %f\n',i-1,100-mean(p==y)*100);
%Sum of each error rate to find the average.
average_error+=100-mean(p==y)*100;
endif
end
%Print the average error rate after cross-validation
fprintf('\n\nAverage Error after cross validation: %f\n\n',average_error/10);
%References
%https://www.gnu.org/software/octave/doc/v4.0.0/
%http://flowingmotion.jojordan.org/2011/10/16/12-steps-to-running-gradient-descent-in-octave/
%https://swizec.com/blog/first-steps-with-octave-and-machine-learning/swizec/2865
%http://blog.madhukaraphatak.com/gradient-descent-for-logistic-regression-in-octave/
%https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
%https://ww2.coastal.edu/kingw/statistics/R-tutorials/logistic.html
%https://www.hackerearth.com/practice/notes/samarthbhargav/logistic-regression-in-apache-spark/
%https://samarthbhargav.wordpress.com/2014/04/22/logistic-regression-in-apache-spark/
%https://en.wikipedia.org/wiki/Logistic_regression
%https://www.gnu.org/software/octave/doc/v4.0.0/Minimizers.html#XREFfminunc