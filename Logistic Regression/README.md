# Description:
Program that implements logistic regression with gradient descent algorithm.<br/><br/>
We have data extracted from images that were taken from genuine and forged banknote-like specimens as training data for logistic regression.<br/>
Data Set Characteristics:  	Multivariate	Number of Instances:	1372<br/>
Attribute Characteristics:	Real	Number of Attributes:	5<br/>
Associated Tasks:	Classification	Missing Values?	No<br/><br/>
Attribute Information:<br/>
1.	variance of Wavelet Transformed image (continuous)<br/>
2.	skewness of Wavelet Transformed image (continuous) <br/>
3.	curtosis of Wavelet Transformed image (continuous)<br/>
4.	entropy of image (continuous) <br/>
5.	class (integer, 0: forged 1: genuine)<br/><br/>
What the code does<br/>
1.	Implements logistic regression with gradient descent<br/>
2.	Uses Cross-Validation to calculate the error rate. In Cross-Validation, randomly split the training data into ratio 90:10. Use the 90% as training data and predict labels on the remaining 10%. Repeat 10 times and average the error rate to get your Cross-Validation estimate.<br/>

Source of the Dataset: http://archive.ics.uci.edu/ml/datasets/banknote+authentication<br/>
Find more datasets on this website: http://archive.ics.uci.edu/ml/datasets.html<br/>

