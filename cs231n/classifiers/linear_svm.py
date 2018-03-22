import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:].transpose() 
        dW[:,y[i]] -= X[i,:].transpose()

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  scores = X.dot(W) #nxc of scores

  train_rows = list(range(num_train))
  correct_class_scores = scores[train_rows, y] #1xn vector of scores for correct class of each nth training example
 
  margins_including_for_correct_classes = (scores.T - correct_class_scores + 1).T #nxc of each margin by class and example
  losses_including_for_correct_classes = np.greater(margins_including_for_correct_classes,0)*margins_including_for_correct_classes
  loss = np.sum(losses_including_for_correct_classes) - num_train
  loss /= num_train
  loss += reg * np.sum(W * W)
  
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
  # loss.                                                                     #
  #############################################################################
  pass
    
  #margins_including_zeroed_out_for_correct_classes = margins_including_for_correct_classes
  #margins_including_zeroed_out_for_correct_classes[train_rows,y] = 0
  #weight_contributions_for_correct_class = np.zeros(margins_including_for_correct_classes.shape)
  #weight_contributions_for_correct_class[train_rows,y] = 1
  #dw = X.transpose().dot(np.greater(margins_including_zeroed_out_for_correct_classes,0)) # dxc \
  #dw -= X.transpose().dot(weight_contributions_for_correct_class)
  #dw /= num_train
  #dW += 2*reg*W
    
  
  #(margins_including_for_correct_classes > 0) is a nxc of booleans for whether class c contributed to nth examaple loss Ln
  #number_of_classes_contributing_to_nth_loss = np.sum(margins_including_for_correct_classes > 0, axis=1) - 1 #1xn of number of classes contributing to nth example loss Ln
  #X is nxd , dW is dxc
    
  margins_including_for_correct_classes[margins_including_for_correct_classes > 0] = 1
  margins_including_for_correct_classes[margins_including_for_correct_classes < 0] = 0
  margins_including_for_correct_classes[range(num_train), y] = 0
 
  row_sums = np.sum(margins_including_for_correct_classes,axis=1) #number of clases contributing to nth example loss Ln

  margins_including_for_correct_classes[range(num_train), y] = -1.0 * row_sums
    
  dW = np.dot(X.T, margins_including_for_correct_classes) #dxn dot product with nxc

  dW /= num_train

  # Add regularization to the gradient
  dW += 2*reg * W
 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  

  return loss, dW
