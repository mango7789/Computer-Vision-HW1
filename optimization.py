from __init__ import *

def get_optim_func(method: str):
    """
    Create a new optimization method based on the input of method. 
    Inputs:
    - method: should be in ['sgd', 'sgd_momentum', 'rmsprop', 'adam'](ignore cases).
    """
    method = method.lower()
    if method == 'sgd':
        return sgd
    elif method == 'sgd_momentum':
        return sgd_momentum
    elif method == 'rmsprop':
        return rmsprop
    elif method == 'adam':
        return adam
    else:
        raise ValueError("Unknown optimization method: {}, please choose from ['sgd', 'sgd_momentum', 'rmsprop', 'adam']".format(method))

def sgd(w: np.array, dw: np.array, config: Dict=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)

    w -= config['learning_rate'] * dw

    return w, config

def sgd_momentum(w: np.array, dw: np.array, config: Dict=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)

    v = config.get('velocity', np.zeros_like(w))
    lr, momentum = config.get('learning_rate'), config.get('momentum')
    v = momentum * v - lr * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config

def rmsprop(w: np.array, dw: np.array, config: Dict=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    decay_rate, cache = config.get('decay_rate'), config.get('cache')
    cache = decay_rate * cache + (1 - decay_rate) * dw * dw
    next_w = w - config.get('learning_rate') / (np.sqrt(cache) + config.get('epsilon')) * dw
    config['cache'] = cache

    return next_w, config

def adam(w: np.array, dw: np.array, config: Dict=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw * dw
    mt = config['m'] / (1 - config['beta1'] ** config['t'])
    vt = config['v'] / (1 - config['beta2'] ** config['t'])
    next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

    return next_w, config 