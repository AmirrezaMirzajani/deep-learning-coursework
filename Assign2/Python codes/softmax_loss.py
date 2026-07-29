#FILE NAME: softmax_loss.py

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, Y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    # 1. Compute unnormalized scores for the i-th sample
    scores = X[i].dot(W)

    # 2. For numerical stability, shift the scores so that the max score is 0
    scores -= np.max(scores)

    # 3. Compute the probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    # 4. Compute the loss for this sample (cross-entropy loss)
    correct_class_prob = probs[Y[i]]
    loss += -np.log(correct_class_prob)

    # 5. Compute the gradient for this sample
    for j in range(num_classes):
      # For the correct class, the gradient is (prob - 1) * input
      if j == Y[i]:
        dW[:, j] += (probs[j] - 1) * X[i]
      # For other classes, the gradient is prob * input
      else:
        dW[:, j] += probs[j] * X[i]

    # 6. Average the loss and gradient over all training examples
  loss /= num_train
  dW /= num_train

  # 7. Add regularization to the loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
    # compute unnormlaized log probs
    # unorm_log_probs = ??

    # for numerical stability in softmax loss add the following line to your code
    # unorm_log_probs -= np.max(unorm_log_probs)

    # get class probabilities
    # compute loss and add it to the loss so far
    # compute gradient and add it to the gradients so far

  # average out grad and loss
  # add regularization loss to total loss and regularization gradienet to dW
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, Y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # 1. Compute scores for the entire batch
  scores = X.dot(W)

  # 2. Numeric stability: shift scores for each sample
  scores -= np.max(scores, axis=1, keepdims=True)

  # 3. Compute probabilities for the entire batch
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # 4. Compute the data loss
  # Select probabilities of the correct classes using numpy indexing
  correct_log_probs = -np.log(probs[range(num_train), Y])
  data_loss = np.sum(correct_log_probs) / num_train

  # 5. Compute the total loss (data loss + regularization loss)
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss

  # 6. Compute the gradient on scores
  dscores = probs
  dscores[range(num_train), Y] -= 1
  dscores /= num_train

  # 7. Backpropagate the gradient to W
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

