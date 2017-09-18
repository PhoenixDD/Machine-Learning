Implements the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.<br/><br/>
1. Training Data<br/>
There are 5000 training examples in trainingdata.mat (or training_x.txt & training_y.txt), where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image. The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index, the digit zero is mapped to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.<br/><br/>
 
Figure 1. Examples from the dataset<br/>
X(5000 x 400) has the form:<br/>
?= [¦(¦(?(x^((1)))?^?@?(x^((2)))?^? )@?@?(x^((m)))?^? )]<br/>
Y(5000 x 1) has the form:<br/>
Y= [¦(¦(y^((1))@y^((2)) )@?@y^((m)) )]<br/><br/>

2. Model representation<br/>
Our neural network is shown in Figure 2. It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20 × 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1). There are 25 units in the second layer and 10 output units (corresponding to the 10 digit classes)<br/><br/>

  
Figure 2: Neural network model<br/>

3. Regularized cost function<br/>
The cost function for neural network with regularization is given by<br/>
J(?)=  1/m ?_(i=1)^m¦?_(k=1)^K¦[-y_k^((i) )  log?((h_? (x^((i) ) ))_k )-(1-y_k^((i) ))log?(1-(h_? (x^((i) ) ))_k ) ] +  ?/2m [?_(j=1)^25¦?_(k=1)^400¦??(??_(j,k)^((1)))?^2 +?_(j=1)^10¦?_(k=1)^25¦??(??_(j,k)^((2)))?^2 ]<br/>
h_? (x^((i) ) ): computed as shown in the Figure 2<br/>
K: the total number of possible labels (K = 10)<br/>
The original labels were 1, 2, …, 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1, so that<br/>
y= [¦(¦(1@0@0)@?@0)],[¦(¦(0@1@0)@?@0)],…or[¦(¦(0@0@0)@?@1)].<br/>
For example, if x(i) is an image of the digit 5, then the corresponding y(i) (that you should use with the cost function) should be a 10-dimensional vector with y5 = 1, and the other elements equal to 0.<br/>
m: the total number of training examples<br/>
?: 1<br/><br/>

4. Random initialization<br/>
While training neural networks, it is important to randomly initialize the parameters for symmetry breaking. One effective strategy for random initialization is to randomly select values for ?(l) uniformly in the range {-?init, ?init}. You should use ?init = 0.12. This range of values ensures that the parameters are kept small and makes the learning more efficient.<br/><br/>
 (T^((1)), T^((2)))<br/>
 epsilon_init=0.12<br/>
 T=rand(rows,columns)*2*epsilon_init-epsilon_init<br/>











5. Backpropagation algorithm
  

PseudoCode:
Load Training set {X y}: {(x(1), y(1)), …, (x(m), y(m))}
Set ?^((1))=zeros ,?^((2))=zeros 
For i = 1 to m
	1. Perform forward propagation to compute for l = 2, 3, 4, …, L
	Set a^((1))  = x^((i))          (add a_0^((1) ))  
	z^((2))  = a^((1) ) ?(T^((1) ))?^T  
	a^((2))=g(z^((2) ) )       (add a_0^((2) ) )    g(x)  is the sigmoid function
	z^((3))  = a^((2)) ?(T^((2) ))?^T  
	a^((3))=g(z^((3) ) )=h_? (x)
	2. Using y^((i)), compute d^((L))=a^((L))-y^((i))
	d^((3))= a^((3))-y
	d^((2))=??(T?^((2)))?^T d^((3)).*g^' (z^((3) ) )                        g^' (x)=g(x)  .*(1-g(x))
	remove the bias row
	3. Update Delta Value
	?^((2))=?^((2))+d^((3)) ?(a^((2) ))?^T
	?^((1))=?^((1))+d^((2) ) ?(a^((1) ))?^T
end for loop
 ?/(?T_ij^((l)) )  J(T)= D_ij^((l))=1/m ?_ij^((l))         for j=0
 ?/(?T_ij^((l)) )  J(T)= D_ij^((l))=1/m ?_ij^((l))+?/m T_ij^((l))         for j>0            ?=1    


6. Gradient descent
Now we have the cost function and partial derivative of the cost function with respect to each of our parameters, we can use gradient descent algorithm to find out parameters theta that minimize the cost function.
Recall gradient descent:
Repeat until convergence {
	Run Backpropagation Algorithm
	T_ij^((l))= T_ij^((l))-a ?/(?T_ij^((l)) )  J(T)
}

7. Prediction & Calculate training accuracy


What the code does
	Follows the instruction to implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.
	Calculates the training set accuracy (the percentage of examples it got correct). The Training accuracy should be greater than 90%, the cost should be less than 1.0

Reference: 
https://www.youtube.com/watch?v=18X68kLAfKY&index=51&list=PLZ9qNFMHZ-A4rycgrgOYma6zxF4BZGGPW
