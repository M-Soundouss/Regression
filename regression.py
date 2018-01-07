# import used libraries
import numpy as np
from sklearn.datasets.samples_generator import make_regression
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


# define loss function
def loss_l2(y_true, y_pred):
    S = np.power((np.array(y_true) - np.array(y_pred)), 2)
    S = np.sum(S)
    return S


# learn linear regression parameters
def linear_regression(x_train, y_train):
    B = np.dot(x_train.transpose(), x_train)
    B = np.linalg.inv(B)
    B = np.dot(B, x_train.transpose())
    B = np.dot(B, y_train)
    return B


if __name__ == '__main__':
    # generate the dataset
    X, y = make_regression(n_samples=200, n_features=5, n_informative=4, n_targets=1, bias=0.0, effective_rank=None,
                           tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
    test_size = 0.20
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size,
                                                                        random_state=seed)
    # add ones to the X_test matrix
    x_nb_rows = X_test.shape[0]
    ones = np.ones((x_nb_rows, 1), dtype=np.int)
    x_test_new = np.append(ones, X_test, axis=1)

    # add ones to the X_train matrix
    xt_nb_rows = X_train.shape[0]
    ones = np.ones((xt_nb_rows, 1), dtype=np.int)
    x_train_new = np.append(ones, X_train, axis=1)

    # learn linear regression parameters using the function defined
    B = linear_regression(x_train_new, Y_train)
    Y_pred = np.dot(x_test_new, B)

    # learn linear regression parameters using sklearn library
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred_sk = lr.predict(X_test)

    # print results
    print("The learned parameters are : %s"%(B))
    print("The loss of the learned parameters is : %s"%(loss_l2(Y_test, Y_pred)))
    print("The loss of the sklearn parameters is : %s"%(loss_l2(Y_test, Y_pred_sk)))

