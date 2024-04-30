from __init__ import *

def get_activation_class(type: str) -> Callable:
    type = type.lower()
    if type == 'relu':
        return ReLU
    elif type == 'tanh':
        return Tanh
    elif type == 'sigmoid':
        return Sigmoid
    else:
        raise ValueError("Unknown activation function: {}, please choose from ['relu', 'tanh', 'sigmoid']".format(type)) 


#######################################################################
#                             ReLU                                    #
#######################################################################
class ReLU():
    @staticmethod
    def forward(x: np.array):
        out = np.copy(x)
        out[out < 0] = 0
        cache = x
        return out, cache
    
    @staticmethod
    def backward(dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = np.copy(dout)
        dx[x <= 0] = 0
        return dx

#######################################################################
#                             Tanh                                    #
#######################################################################
class Tanh():
    @staticmethod
    def forward(x: np.array):
        out = np.tanh(np.copy(x))
        cache = x
        return out, cache

    @staticmethod
    def backward(dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = dout * (1 - np.power(np.tanh(x), 2))
        return dx

#######################################################################
#                            Sigmoid                                  #
#######################################################################
class Sigmoid():
    @staticmethod
    def sigmoid(x: np.array):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def forward(x: np.array):
        out = np.copy(x)
        out = Sigmoid.sigmoid(out)
        cache = x
        return out, cache
    
    @staticmethod
    def backward(dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = dout * Sigmoid.sigmoid(x) * (1 - Sigmoid.sigmoid(x))
        return dx