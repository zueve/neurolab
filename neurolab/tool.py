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
    from cPickle import load

    with open(fname, 'r') as file:
        net = load(file)

    return net

def save(net, fname):
    from cPickle import dump

    with open(fname, 'w') as file:
        dump(net, file)

#------------------------------------------------------------

def np_size(net):
    """
    Calculete count of al network parameters (weight, bias, ect...)

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
    Get all network parameters in one array as referance
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
    for feed-forward neuran networks on each step

    :Parametrs:
        net: Net
            Feed-forward network
        inp: array, size = net.ci
            Input array
        tar: array, size = net.co
            Train target
        deriv: callable
            Derivative of error function
        grad:list of dict default(None)
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
    e = out - tar
    # for output layer
    ln = len(net.layers) - 1
    layer = net.layers[ln]
    delt[ln] = net.errorf.deriv(e) * layer.transf.deriv(layer.s, out)
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
    for feed-forward neuran networks on each step

    :Parametrs:
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
            Dradient of net for each layer,
            format:[{'w':..., 'b':...},{'w':..., 'b':...},...]
        grad_flat: array
            All neurons propertys in 1 array (reference of grad)
        output: array
            output of network

    """
    grad_flat = np.zeros(np_size(net))
    grad = []
    st = 0
    for i, l in enumerate(net.layers):
        grad.append({})
        for k, v in l.np.items():
            grad[i][k] = grad_flat[st: st + v.size]
            grad[i][k].shape = v.shape
            st += v.size
    output = []
    for inp, tar in zip(input, target):
        out = net.step(inp)
        ff_grad_step(net, out, tar, grad)
        output.append(out)
    return grad, grad_flat, np.row_stack(output)


def simhop(net, input, n=10):
    """
    Simuate hopfied network

	OLD VERSION, now you may use newhop new with native sim method
	This function may be deleted in future (use newhop(...).sim())

    :Parameters:
        net: Net
            Simulated recurrent neural network like Hopfield (newhop_old only)
        input: array like (N x net.ci)
            Train input patterns
        n: int (default 10)
            Maximum number of simulated steps

    :Return:
        output: array
            Network outputs
        full_output: list of array
            Network outputs, including the intermediate results
    :Exmamle:
        >>> from .net import newhop_old
        >>> target = [[-1, -1, -1], [1, -1, 1]]
        >>> net = newhop_old(target)
        >>> simhop(net, target)[0]
        array([[-1., -1., -1.],
               [ 1., -1.,  1.]])

    """

    input = np.asfarray(input)

    assert input.ndim == 2
    assert input.shape[1] == net.layers[-1].co
    assert input.shape[1] == net.ci

    output = []
    for inp in input:
        net.layers[-1].out = inp
        out = []
        for i in range(n):
            o = net.step(inp)
            if i>0 and np.all(out[-1] == o):
                break
            out.append(o)
        output.append(np.array(out))
    return np.array([r[-1] for r in output]), output
