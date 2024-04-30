from __init__ import *

def ce_multi_loss(x: np.array, y: np.array):
	"""
	Computes the cross-entrophy loss and gradient for multi-class classification.
	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for
	  the jth class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label
	  for x[i] and 0 <= y[i] < C
	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	# use shifted_logits to avoid overflow
	shifted_logits = x - np.max(x, axis=1, keepdims=True)    
	Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
	log_probs = shifted_logits - np.log(Z)

	probs = np.exp(log_probs)
	N = x.shape[0]
	loss = (-1.0 / N) * np.sum(log_probs[np.arange(N), y])
	dx = np.copy(probs)
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx

