#FILE NAME: svm_loss.py

import numpy as np
from random import shuffle

def svm_loss_naive(W, X, Y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - Y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[Y[i]]
    for j in range(num_classes):
      if j == Y[i]:
        continue   #ignore correct class
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of margin and add it to corresponding elements in dW #
        #############################################################################
        # For the incorrect class, gradient is +X[i]
        dW[:, j] += X[i]
        # For the correct class, gradient is -X[i]
        dW[:, Y[i]] -= X[i]
        #############################################################################
        #              END OF YOUR CODE                                             #
        #############################################################################

  #################################################################################
  # TODO:                                                                         #
  # Right now the loss and the gradient is a sum over all training examples, but  #
  # we want it to be an average instead,                                          #
  # So average out grad and loss by dividing by num_train.                        #
  #                                                                               #
  # Then, add regularization loss to the loss and                                 #
  # regularization gradienet to dW
  #
    # Average the loss and gradient
  loss /= num_train
  dW /= num_train

    # Add regularization to the loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #################################################################################

  return loss, dW


def svm_loss_vectorized(W, X, Y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute scores  (N x C)
  # grab correct class scores
  # compute margins
  # add reg loss to the loss

  # 1. Compute scores
  scores = X.dot(W)

  # 2. Get correct class scores and compute margins
  correct_class_scores = scores[np.arange(num_train), Y].reshape(num_train, 1)
  margins = np.maximum(0, scores - correct_class_scores + 1)

  # 3. Set margin for correct classes to 0
  margins[np.arange(num_train), Y] = 0

  # 4. Compute data loss and add regularization
  loss = np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss (e.g. margins).                                                      #
  #############################################################################
  # 1. Create a binary mask for positive margins
  binary_mask = np.zeros(margins.shape)
  binary_mask[margins > 0] = 1

  # 2. For each sample, count how many classes had a positive margin
  # and set the gradient for the correct class to the negative of this count
  row_sum = np.sum(binary_mask, axis=1)
  binary_mask[np.arange(num_train), Y] = -row_sum

  # 3. Compute the gradient with respect to weights
  dW = X.T.dot(binary_mask)

  # 4. Average the gradient and add regularization gradient
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
