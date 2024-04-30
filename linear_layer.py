from __init__ import *
from activation_func import get_activation_class


class Linear:
	@staticmethod
	def forward(x: np.array, w: np.array, b: np.array):
		"""
		Computes the forward pass for an linear (fully-connected) layer.
		The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
		examples, where each example x[i] has shape (d_1, ..., d_k). We will
		reshape each input into a vector of dimension D = d_1 * ... * d_k, and
		then transform it to an output vector of dimension M.
		Inputs:
		- x: An array containing input data, of shape (N, d_1, ..., d_k)
		- w: An array of weights, of shape (D, M)
		- b: An array of biases, of shape (M,)
		Returns a tuple of:
		- out: output, of shape (N, M)
		- cache: (x, w, b), used for backward pass
		"""
		N = x.shape[0]
		reshape_x = x.reshape(N, -1)
		out = np.matmul(reshape_x, w) + b
		cache = (x, w, b)
		return out, cache
	
	@staticmethod
	def backward(dout: np.array, cache: tuple):
		"""
		Computes the backward pass for an linear layer.
		Inputs:
		- dout: Upstream derivative, of shape (N, M)
		- cache: Tuple of:
		- x: Input data, of shape (N, d_1, ... d_k)
		- w: Weights, of shape (D, M)
		- b: Biases, of shape (M,)
		Returns a tuple of:
		- dx: Gradient with respect to x, of shape
			(N, d1, ..., d_k)
		- dw: Gradient with respect to w, of shape (D, M)
		- db: Gradient with respect to b, of shape (M,)
		"""
		x, w, b = cache
		dx = np.matmul(dout, np.transpose(w)).reshape(x.shape)
		reshape_x = x.reshape(x.shape[0], -1)
		dw = np.matmul(np.transpose(reshape_x), dout)
		db = np.sum(dout, axis=0)
		return dx, dw, db
		
class LinearActivation:
	@staticmethod
	def forward(x: np.array, w: np.array, b: np.array, type: str):
		"""
		Convenience layer that performs an linear transform
		followed by a activation function.

		Inputs:
		- x: Input to the linear layer
		- w, b: Weights for the linear layer
		- type: The activation function to use
		Returns a tuple of:
		- out: Output from the activation function
		- cache: Object to give to the backward pass
		"""
		score, fc_cache = Linear.forward(x, w, b)
		out, af_cache = get_activation_class(type).forward(score)
		cache = (fc_cache, af_cache)
		return out, cache
	
	@staticmethod
	def backward(dout: np.array, cache: tuple, type: str):
		"""
		Backward pass for the linear-activation_function convenience layer
		"""
		fc_cache, af_cache = cache
		ds = get_activation_class(type).backward(dout, af_cache)
		dx, dw, db = Linear.backward(ds, fc_cache)
		return dx, dw, db
