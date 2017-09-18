# Description:
Bias-Variance Tradeoff and Learning Curve <br/><br/>

An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data. In this programming assignment, this will plot training and test errors on a learning curve to diagnose bias-variance problems.<br/><br/>

We have a dataset (Figure 1) containing historical records on the change in the water level, x, and the amount of water flowing out of the dam, y.<br/>
This dataset is divided into 2 parts:<br/>
A training set that your model will learn on: training_x, training_y<br/>
A cross validation set for calculate cross validation error: cv_x, cv_y<br/><br/>

 

PART 1:<br/>
For linear regression (degree = 1):<br/>
our hypothesis has the form:<br/>
h_? (x)=?_0+?_1 x_1<br/>
Applies linear regression to our training data to get ? parameter. Then computes the error on the training and cross validation sets. Note that the training error for a dataset is defined as<br/><br/>
Err(?)=1/2m [?_(i=1)^m¦(h_? (x^((i) ) )-y(i))^2 ]<br/>

To plot the learning curve, we need a training and cross validation set error for different training set sizes.<br/><br/>
As we have 12 rows of training data set, to obtain error with different training set sizes, we should repeat 11 times linear regression with different subsets of the original training set. When we are computing the training set error, we make sure that we compute it on the training subset.<br/><br/>
For instance:<br/>
Round 1: uses training subset (row: 1, 2) to train your linear regression model. Computes the training error on the training subset. Computes the cross-validation error over the entire cross validation set.<br/>
Round 2: uses training subset (row: 1, 2, 3)<br/>
Round 3: uses training subset (row: 1, 2, 3, 4)<br/>
…<br/>
 At last, we will produce a plot similar to Figure 2.<br/><br/>
 
In Figure 2, we observe that both the training error and cross validation error are high when the number of training examples is increased. This reflects a high bias problem in the model - the linear regression model is too simple and is unable to fit our dataset well.<br/><br/>

PART 2:<br/>
The problem with our linear model was that it was too simple for the data and resulted in under fitting (high bias). In this part we will address this problem by adding more features. <br/>
For use polynomial regression (degree = p, p > 1), our hypothesis has the form:<br/>
h_? (x)=?_0+?_1 x+?_2 x^2+?_3 x^3+?+?_p x^p<br/>
If the degree is too high, we will observe in learning curve figure that low training error is low, but the cross validation error is high. There is a gap between the training and cross validation errors, indicating a high variance problem.<br/><br/>

The program follows the above instructions, generates the learning curve with degree 1 and degree 6.<br/>

Reference: <br/>
https://www.youtube.com/watch?v=Iz61geqni7g&index=59&list=PLJ1-ciQ35nuiyL1PX6O4NdF5CjjaDdnVC<br/>

