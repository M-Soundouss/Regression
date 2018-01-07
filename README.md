# Regression

The objective of this repository is to implement linear and logistic regression from scratch in Python.

## Linear regression
First, I started by defining the loss function and the function for learning the linear regression parameters.

Second, I generated a regression dataset using sklearn with 200 samples, 5 features from which 4 are informative and with no bias. This dataset was divided into train and test sets with a test size of 20%.

Third, I added and array of ones in the beginning of the train and test sets. Then I learned the parameters with the defined function and sklearn linear regression and used both of them to predict Y test values.

Finally, I printed the results. The first element in the learned parameters' array represents the bias which is close to zero and reflects the bias chosen in the dataset generation. The other elements represent the values of the parameters for each feature. Only one of them has a value close to zero since I chose that 4 out of the 5 features are informative. The losses of the learned parameters and the sklearn parameters are very close, reflecting the correctness of the implemented linear regression.