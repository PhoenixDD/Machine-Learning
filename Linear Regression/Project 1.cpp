/***********************************************************************************************************************************************************
    Author : Dhairya Dhondiyal

    Description:
    The Program Calculates the Ordinary Least squared vector and stores in (X_Xt_Xt) to find the predicted values.
    The test data is multiplied by the OLS vector to get the predicted values.
    OLS=(X_t*X)*X_t*Label_t
    _t=Transpose.

    This program is explicitly made for the Given training data.
    The training data is Hardcoded into the program rather than dynamically taking from file.

    NOTE: To change Training data the first few lines of main() needs to be changed along with the number of attributes.
	
    Disclaimer:
    Copyright (C) - All Rights Reserved
    Unauthorized copying of this file, via any medium is strictly prohibited
    Proprietary and confidential
    Written by Dhairya Dhondiyal, September 2016

************************************************************************************************************************************************************/
#include<iostream>
#include<fstream>                                       //For file output.
#include<math.h>                                        //For Power function.
using namespace std;
#define attributes 7                                    //Number of attributes/features in dataset(# of columns), Change according to the problem.
                                                        //NOTE: # of features has to be changes explicitly. Change if number of features change.

void cofactor(double matrix[attributes][attributes],double temp[attributes][attributes],int p,int q,int n)
{                                                                                                          //}
    int a=0,b=0;                                                                                           //}
    for(int i=0;i<n;i++)                                                                                   //}
    {                                                                                                      //}
        for(int j=0;j<n;j++)                                                                               //}
        {                                                                                                  //}
            if(i!=p&&j!=q)                                                                                 //}      Finds the Minors and Cofactors of the Matrix.
            {                                                                                              //}
                temp[a][b++]=matrix[i][j];                                                                 //}
                if(b==n-1)                                                                                 //}
                {                                                                                          //}
                    b=0;                                                                                   //}
                    a++;                                                                                   //}
                }                                                                                          //}
            }                                                                                              //}
        }                                                                                                  //}
    }                                                                                                      //}
}                                                                                                          //}
double determinant(double matrix[attributes][attributes],int dim)
{
	double sum=0,c[attributes][attributes];                                                                //}
	if(dim==1)                                                                                             //}
        return matrix[0][0];                                                                               //}
	for(int p=0;p<dim;p++)                                                                                 //}
	{                                                                                                      //}
		int h=0,k=0;                                                                                       //}
		for(int i=1;i<dim;i++)                                                                             //}
		{                                                                                                  //}
			for(int j=0;j<dim;j++)                                                                         //}
			{                                                                                              //}
				if(j==p)                                                                                   //}
				continue;                                                                                  //}
				c[h][k]=matrix[i][j];                                                                      //}
				k++;                                                                                       //}       Calculates the Determinant of the matrix using recursion.
				if(k==dim-1)                                                                               //}
				{                                                                                          //}
					h++;                                                                                   //}
					k=0;                                                                                   //}
				}                                                                                          //}
			}                                                                                              //}
		}                                                                                                  //}
		sum=sum+matrix[0][p]*pow(-1,p)*determinant(c,dim-1);                                               //}
	}                                                                                                      //}
	return sum;                                                                                            //}
}                                                                                                          //}
void adjoint(double matrix[attributes][attributes],double adj[attributes][attributes])
{                                                                                                          //}
    int sign=1;                                                                                            //}
    double cofct[attributes][attributes];                                                                  //}
    for(int i=0;i<attributes;i++)                                                                          //}
    {                                                                                                      //}
        for(int j=0;j<attributes;j++)                                                                      //}   Adjoint of the Matrix using Cofactors and transpose.
        {                                                                                                  //}
            cofactor(matrix,cofct,i,j,attributes);                                                         //}
            sign=((i+j)%2==0)?1:-1;                                                                        //}
            adj[j][i]=(sign)*(determinant(cofct,attributes-1));                                            //}
        }                                                                                                  //}
    }                                                                                                      //}
}                                                                                                          //}
void inv(double matrix[attributes][attributes],double inverse[attributes][attributes])
{                                                                                                          //}
    double det=determinant(matrix, attributes);                                                            //}
    double adj[attributes][attributes];                                                                    //}
    adjoint(matrix,adj);                                                                                   //}      Inverse of a matrix using Adjoint and determinant.
    for(int i=0; i<attributes; i++)                                                                        //}
        for(int j=0;j<attributes;j++)                                                                      //}
            inverse[i][j]=adj[i][j]/det;                                                                   //}
}                                                                                                          //}
int main()
{

    //This program is explicitly made for the following training data.
    //The training data is Hardcoded into the program rather than dynamically taking from file.
    //NOTE: To change Training data the first few lines needs to be changed along with the number of attributes.

    //Label (Feature 1)(Feature to be predicted)Total overall reported crime rate per 1 million residents.
    double label[]={478,494,643,341,773,603,484,546,424,548,506,819,541,491,514,371,457,437,570,432,619,357,623,547,792,799,439,867,912,462,859,805,652,776,919,732,657,1419,989,821,1740,815,760,936,863,783,715,1504,1324,940};
    //Feature 2 (Reported violent crime rate per 100,000 residents).
    double cr100[]={184,213,347,565,327,260,325,102,38,226,137,369,109,809,29,245,118,148,387,98,608,218,254,697,827,693,448,942,1017,216,673,989,630,404,692,1517,879,631,1375,1139,3545,706,451,433,601,1024,457,1441,1022,1244};
    //Feature 3 (Annual police funding in $/resident).
    double pf[]={40,32,57,31,67,25,34,33,36,31,35,30,44,32,30,16,29,36,30,23,33,35,38,44,28,35,31,39,27,36,38,46,29,32,39,44,33,43,22,30,86,30,32,43,20,55,44,37,82,66};
    //Feature 4 (% of people 25 years+ with 4 yrs. of high school).
    double fouryr_25_hiscl[]={74,72,70,71,72,68,68,62,69,66,60,81,66,67,65,64,64,62,59,56,46,54,54,45,57,57,61,52,44,43,48,57,47,50,48,49,72,59,49,54,62,47,45,48,69,42,49,57,72,67};
    //Feature 5 (% of 16 to 19 year-olds not in high school and not high school graduates).
    double no_16_19_hiscl[]={11,11,18,11,9,8,12,13,7,9,13,4,9,11,12,10,12,7,15,15,22,14,20,26,12,9,19,17,21,18,19,14,19,19,16,13,13,14,9,13,22,17,34,26,23,23,18,15,22,26};
    //Feature 6 (% of 18 to 24 year-olds in college).
    double coll_18_24[]={31,43,16,25,29,32,24,28,25,58,21,77,37,37,35,42,21,81,31,50,24,27,22,18,23,60,14,31,24,23,22,25,25,21,32,31,13,21,46,27,18,39,15,23,7,23,30,35,15,18};
    //Feature 7 (% of 18 to 24 year-olds in college).
    double plus25_coll[]={20,18,16,19,24,15,14,11,12,15,9,36,12,16,11,14,10,27,16,15,8,13,11,8,11,18,12,10,9,8,10,12,9,9,11,14,22,13,13,12,15,11,10,12,12,11,12,13,16,16};
    int tot_columns=sizeof(label)/sizeof(label[0]);                                  //Calculate size of each feature(#Rows of each feature)
    double label_t[tot_columns][1];                                                  //Store Transpose of Label.

    for(int i=0;i<tot_columns;i++)                                                   //}
        for(int j=0;j<1;j++)                                                         //}   To Transpose Label and store.
            label_t[i][j]=label[i];                                                  //}

    double X[tot_columns][attributes];                                               //   Store Matrix of Training data

    for(int i=0;i<tot_columns;i++)                                                   //}
    {                                                                                //}
        for(int j=0;j<attributes;j++)                                                //}
            if(j==0)                                                                 //}
                X[i][j]=1;                                                           //}
            else if(j==1)                                                            //}   Create Matrix of features as training data.(X)
                X[i][j]=cr100[i];                                                    //}
            else if(j==2)                                                            //}   NOTE: Explicitly add all the features. Change if # of features change in training data.
                X[i][j]=pf[i];                                                       //}
            else if(j==3)                                                            //}
                X[i][j]=fouryr_25_hiscl[i];                                          //}
            else if(j==4)                                                            //}
                X[i][j]=no_16_19_hiscl[i];                                           //}
            else if(j==5)                                                            //}
                X[i][j]=coll_18_24[i];                                               //}
            else if(j==6)                                                            //}
                X[i][j]=plus25_coll[i];                                              //}
    }                                                                                //}

    double Xt[attributes][tot_columns];                                              //Store Transpose of Matrix of features (X_t)

    for(int i=0;i<attributes;i++)                                                    //}
        for(int j=0;j<tot_columns;j++)                                               //} To transpose Matrix of features and store(X_t).
            Xt[i][j]=X[j][i];                                                        //}

    double Xt_X[attributes][attributes];                                             // Store Transpose of Matrix of features x Matrix of features.(X_t*X)

    for(int i=0;i<attributes;i++)                                                    //}
        for(int j=0;j<attributes;j++)                                                //}
        {                                                                            //}
            Xt_X[i][j]=0;                                                            //}
            for(int k=0;k<tot_columns;k++)                                           //}   For Multiplication of Transpose of Matrix of features and Matrix of features.(X_t*X)
            {                                                                        //}
                Xt_X[i][j]+=Xt[i][k]*X[k][j];                                        //}
            }                                                                        //}
        }                                                                            //}

    double inverse[attributes][attributes];                                          //Store inverse of Multiplication of Transpose of Matrix of features and Matrix of features.(X_t*X)^-1

    inv(Xt_X,inverse);                                                               //Call function to inverse the Matrix.

    double Xt_X_Xt[attributes][tot_columns];                                         //Store multiplication of Previous multiplied Matrices and transpose of matrix of features.(X_t*X)^-1*X_t

    for(int i=0;i<attributes;i++)                                                    //}
        for(int j=0;j<tot_columns;j++)                                               //}
        {                                                                            //}
            Xt_X_Xt[i][j]=0;                                                         //}  Multiply the Previous multiplied Matrices and transpose of matrix of features.(X_t*X)^-1*X_t
            for(int k=0;k<attributes;k++)                                            //}
            {                                                                        //}
                Xt_X_Xt[i][j]+=inverse[i][k]*Xt[k][j];                               //}
            }                                                                        //}
        }                                                                            //}

    double Xt_X_Xt_label_t[attributes][1];                                           //Store Multiplication of Previous multiplied Matrices and Transpose of label matrix.(X_t*X)^-1*X_t*Label_t Ordinary least squared column vector

    for(int i=0;i<attributes;i++)                                                    //
        for(int j=0;j<1;j++)                                                         //
        {                                                                            //    Final Ordinary Least squared column vector.
            Xt_X_Xt_label_t[i][j]=0;                                                 //
            for(int k=0;k<tot_columns;k++)                                           //     Multiply Previous multiplied Matrices and Transpose of label matrix.(X_t*X)^-1*X_t*Label_t
            {                                                                        //
                Xt_X_Xt_label_t[i][j]+=Xt_X_Xt[i][k]*label_t[k][j];                  //
            }                                                                        //
        }                                                                            //

    double test[][attributes]={{1,999 ,33,50,12,55,30},                              //
                               {1,576 ,76,20,21,43,60},                              //
                               {1,2304,92,81,32,22,10},                              //
                               {1,382 ,27,74,17,30,22},                              //
                               {1,738 ,37,67,8 ,33,13},                              // Test data Matrix. Columns=Features, Rows=# of tests/Prediction
                               {1,1723,33,51,25,17,28},                              //
                               {1,827 ,73,78,21,18,31},                              //
                               {1,182 ,29,41,31,21,19},                              //
                               {1,382 ,47,39,9 ,8 ,7 },                              //
                               {1,1129,49,48,11,9,12}};                              //

    double predicted_values[tot_columns][1];                                         // Store Answer. (Column Matrix for Predicted values).

    int test_rows=sizeof(test)/sizeof(test[0]);                                      // Store Calculated # of tests.(Rows of test data).

    for(int i=0;i<test_rows;i++)                                                     //}
        for(int j=0;j<1;j++)                                                         //}
        {                                                                            //}
            predicted_values[i][j]=0;                                                //}
            for(int k=0;k<attributes;k++)                                            //}  Multiply Ordinary least Squared column vector with Test data matrix to obtain predicted values.
            {                                                                        //}
                predicted_values[i][j]+=test[i][k]*Xt_X_Xt_label_t[k][j];            //}
            }                                                                        //}
        }                                                                            //}

    ofstream output("output.txt");                                                   //   Open Output stream to write into "output.txt" file

    for(int i=0;i<test_rows;i++)                                                     //}
    {                                                                                //}
        for(int j=0;j<1;j++)                                                         //}  Output the Predicted values through Predicted column vector
            cout<<predicted_values[i][j];                                            //}
        cout<<endl;                                                                  //}
    }                                                                                //}

    for(int i=0;i<test_rows;i++)                                                               //}
    {                                                                                          //}
        for(int j=0;j<1;j++)                                                                   //}
            output<<"Predicted Crime Rate for #"<<i+1<<" row:\t"<<predicted_values[i][j]<<"\n";//} Send Formatted output data to "output" file.
        cout<<endl;                                                                            //}
    }                                                                                          //}

    output.close();                                                                            // Close the output stream.
    return 0;
}
