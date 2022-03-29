import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    x_w = np.dot(np.array(X), w)
    squared_error = np.square(np.subtract(x_w, y))
    err = np.mean(squared_error, dtype=np.float64)
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################

  xt_x = np.dot(np.array(X).transpose(), np.array(X))
  w = np.dot(np.dot(np.linalg.inv(xt_x), np.array(X).transpose()), np.array(y))
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################

    xt = np.array(X).transpose()
    xt_x = np.dot(xt, np.array(X))
    l_i = lambd * np.identity(np.size(xt_x, 1))
    xt_x_plus_li = np.add(xt_x, l_i)
    xt_y = np.dot(xt, y)
    w = np.dot(np.linalg.inv(xt_x_plus_li), xt_y)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################

    bestlambda = None
    min_mse = float("inf")
    for i in range(-14, 1):
        w = regularized_linear_regression(Xtrain, ytrain, 2 ** i)
        mse_test = mean_square_error(w, Xval, yval)
        if mse_test < min_mse:
            min_mse = mse_test
            bestlambda = 2 ** i

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    x_temp = X
    for power in range(2, p + 1):
        X = np.concatenate((X, np.power(x_temp, power)), axis=1)
    return X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

