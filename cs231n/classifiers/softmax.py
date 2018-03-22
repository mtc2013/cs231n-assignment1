import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores) #NxC array of scores for each nth example
   
  for n in xrange(X.shape[0]):
      correct_class = y[n] 
      sum = 0.0
      for c in xrange(W.shape[1]):
        if c == correct_class:
            loss -= scores[n,c]
        sum += np.exp(scores[n,c])
      loss += np.log(sum)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)   
    

  for c in xrange(W.shape[1]):
      for n in xrange(X.shape[0]):
        scaling_factor = 1/np.sum(np.exp(scores[n,:]))
        dW[:,c] += scaling_factor * np.exp(scores[n,c])*X[n,:]
        if y[n] == c:
          dW[:,c] += -1.0*X[n,:]
                
  dW /= X.shape[0]              
  dW += 2*reg*W
            
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  scores = X.dot(W)
  scores -= np.max(scores) #NxC array of scores for each nth example

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss += -1.0*np.sum(scores[xrange(X.shape[0]),y])  
  loss += np.sum(np.log(np.sum(np.exp(scores),axis=1)))
    
  loss /= X.shape[0]
  loss += reg * np.sum(W * W) 
    
  
  mask = np.zeros((X.shape[0],W.shape[1])) #nxc matrix 
  mask[xrange(X.shape[0]), y] = 1 

  dW += -1.0 * X.transpose().dot(mask)
    
  m = (1/np.sum(np.exp(scores), axis=1)) * X.transpose() #array of n summed exponentiated scores multiplied broadcasted with the data to give dxn matrix 
  dW += m.dot(np.exp(scores))#nxc matric of exponentiated score for class
  

  dW /= X.shape[0]
  dW += 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

