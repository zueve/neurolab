# -*- coding: utf-8 -*-
"""
Some helpful tools

"""
import numpy as np


def minmax(input):
    """
    Calculate min, max for each row

    """
    input = np.asfarray(input)
    assert input.ndim == 2
    min = input.min(axis=0)
    max = input.max(axis=0)
    out = [x for x in zip(min, max)]
    return tuple(out)


class Norm:
    def __init__(self, x):

        x = np.asfarray(x)
        if x.ndim != 2:
            raise ValueError('x mast have 2 dimensions')
        min = np.min(x, axis=0)
        dist = np.max(x, axis=0) - min

        min.shape = 1, min.size
        dist.shape = 1, dist.size

        self.min = min
        self.dist = dist

    def __call__(self, x):
        x = np.asfarray(x)
        res = (x - self.min) / self.dist

        return res

    def renorm(self, x):
        x = np.asfarray(x)

        res = x * self.dist + self.min
        return res

#------------------------------------------------------------

def load(fname):
    from pickle import load

    with open(fname, 'rb') as file:
        net = load(file)

    return net

def save(net, fname):
    from pickle import dump

    with open(fname, 'wb') as file:
        dump(net, file)

#------------------------------------------------------------

def np_size(net):
    """
    Calculate count of al network parameters (weight, bias, etc...)

    """

    size = 0
    for l in net.layers:
        for prop in l.np.values():
            size += prop.size
    return size

def np_get(net):
    """
    Get all network parameters in one array

    """
    size = np_size(net)
    result = np.zeros(size)
    start = 0
    for l in net.layers:
        for prop in l.np.values():
            result[start: start+prop.size] = prop.flat[:]
            start += prop.size
    return result

def np_set(net, np_data):
    """
    Set network parameters

    :Example:
    >>> import neurolab as nl
    >>> net = nl.net.newff([[-1, 1]], [3, 1])
    >>> x = np_get(net)
    >>> x.fill(100)
    >>> np_set(net, x)
    >>> net.layers[0].np['w'].tolist()
    [[100.0], [100.0], [100.0]]

    """
    start = 0
    for l in net.layers:
        for prop in l.np:
            size = l.np[prop].size
            values = np_data[start: start+size]
            values.shape = l.np[prop].shape
            l.np[prop][:] = values
            start += size

def np_get_ref(net):
    """
    Get all network parameters in one array as reference
    Change array -> change networks

    :Example:
    >>> import neurolab as nl
    >>> net = nl.net.newff([[-1, 1]], [3, 1])
    >>> x = np_get_ref(net)
    >>> x.fill(10)
    >>> net.layers[0].np['w'].tolist()
    [[10.0], [10.0], [10.0]]

    """
    size = np_size(net)
    x = np.empty(size)
    st = 0
    for l in net.layers:
        for k, v in l.np.items():
            x[st: st + v.size] = v.flatten()
            l.np[k] = x[st: st + v.size]
            l.np[k].shape = v.shape
            st += v.size
    return x

#------------------------------------------------------------

def ff_grad_step(net, out, tar, grad=None):
    """
    Calc gradient with backpropogete method,
    for feed-forward neuron networks on each step

    :Parametrs:
        net: Net
            Feed-forward network
        inp: array, size = net.ci
            Input array
        tar: array, size = net.co
            Train target
        deriv: callable
            Derivative of error function
        grad: list of dict default(None)
            Grad on previous step
    :Returns:
        grad: list of dict
            Gradient of net for each layer,
            format:[{'w':..., 'b':...},{'w':..., 'b':...},...]

    """
    delt = [None] * len(net.layers)
    if grad is None:
        grad = []
        for i, l in enumerate(net.layers):
            grad.append({})
            for k, v in l.np.items():
                grad[i][k] = np.zeros(v.shape)

    # for output layer
    ln = len(net.layers) - 1
    layer = net.layers[ln]
    delt[ln] = net.errorf.deriv(tar, out) * layer.transf.deriv(layer.s, out)
    delt[ln] = np.negative(delt[ln])
    delt[ln].shape = delt[ln].size, 1
    grad[ln]['w'] += delt[ln] * layer.inp
    grad[ln]['b'] += delt[ln].reshape(delt[ln].size)

    bp = range(len(net.layers) -2, -1, -1)
    for ln in bp:
        layer = net.layers[ln]
        next = ln + 1

        dS = np.sum(net.layers[next].np['w'] * delt[next], axis=0)
        delt[ln] = dS * layer.transf.deriv(layer.s, layer.out)
        delt[ln].shape = delt[ln].size, 1

        grad[ln]['w'] += delt[ln] * layer.inp
        grad[ln]['b'] += delt[ln].reshape(delt[ln].size)
    return grad


def ff_grad(net, input, target):
    """
    Calc and accumulate gradient with backpropogete method,
    for feed-forward neuron networks on each step

    :Parameters:
        net: Net
            Feed-forward network
        input: array, shape = N,net.ci
            Input array
        target: array, shape = N,net.co
            Train target
        deriv: callable
            Derivative of error function
    :Returns:
        grad: list of dict
            Gradient of net for each layer,
            format:[{'w':..., 'b':...},{'w':..., 'b':...},...]
        grad_flat: array
            All neurons property's in 1 array (reference of grad)
            It link to grad (changes grad is changes grad_flat)
        output: array
            output of network

    """
    # Init grad and link to grad_falt
    grad = []
    grad_flat = np.zeros(np_size(net))
    st = 0
    for i, l in enumerate(net.layers):
        grad.append({})
        for k, v in l.np.items():
            grad[i][k] = grad_flat[st: st + v.size]
            grad[i][k].shape = v.shape
            st += v.size
    output = []
    # Calculate grad for all batch
    for inp, tar in zip(input, target):
        out = net.step(inp)
        ff_grad_step(net, out, tar, grad)
        output.append(out)
    return grad, grad_flat, np.row_stack(output)


def reg_norms(net, ord=2):
    """
    Calculate norm of weights and and biases for calculating
    the regularization term.
    :Parameters:
        net: neurolab net object
    :Keywords:
        ord: int
            order of norm for regularization term. Usually in {1,2}
    
    """

    # Assemble weights and biases into 1D vectors
    w = []
    b = []
    for layer in net.layers:
        w.extend(layer.np['w'].reshape(layer.np['w'].size))
        b.extend(layer.np['b'].reshape(layer.np['b'].size))

    # Calculate norms 
    w = np.linalg.norm(w, ord=ord)
    b = np.linalg.norm(b, ord=ord)

    return w, b

def reg_error(e, net, rr):
    """
    Apply regularization for result to error function
    
    :Parameters:
        e: float
            current error position
        net: neurolab net object
        rr: float
            regularization rate [0, 1]
    :Return:
        output: array
        Gradient with regularization
        
    """
    
    w, b = reg_norms(net)
    e += rr * w + rr * b
    return e

def reg_grad(grad, net, rr):
    """
    Correction gradient for regularization
    
    :Parameters:
        grad: list of dict
            grad without regularization
        net: neurolab net object
        rr: float
            regularization rate [0, 1]
    :Return:
        output: array
        Gradient with regularization
        
    """
    for i, l in enumerate(net.layers):
        grad[i]['w'] += rr * l.np['w']
        grad[i]['b'] += rr * l.np['b']
    return grad
