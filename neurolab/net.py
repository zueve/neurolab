# -*- coding: utf-8 -*-
"""
The module contains the basic network architectures


+-------------------------+------------+---------+-----------------+----------+
|      Network Type       |  Function  | Count of|Support train fcn| Error fcn|
|                         |            | layers  |                 |          |
+=========================+============+=========+=================+==========+
| Single-layer perceptron |    newp    |    1    |   train_delta   |   SSE    |
+-------------------------+------------+---------+-----------------+----------+
| Multi-layer perceptron  |   newff    |  >=1    |   train_gd,     |   SSE    |
|                         |            |         |   train_gdm,    |          |
|                         |            |         |   train_gda,    |          |
|                         |            |         |   train_gdx,    |          |
|                         |            |         |   train_rprop,  |          |
|                         |            |         |   train_bfgs*,  |          |
|                         |            |         |   train_cg      |          |
+-------------------------+------------+---------+-----------------+----------+
|    Competitive layer    |    newc    |    1    |   train_wta,    |   SAE    |
|                         |            |         |   train_cwta*   |          |
+-------------------------+------------+---------+-----------------+----------+
|           LVQ           |   newlvq   |    2    |   train_lvq     |   MSE    |
+-------------------------+------------+---------+-----------------+----------+
|           Elman         |   newelm   |  >=1    |   train_gdx     |   MSE    |
+-------------------------+------------+---------+-----------------+----------+
|         Hopield         |   newhop   |    1    |       None      |   None   |
+-------------------------+------------+---------+-----------------+----------+
|         Hemming         |   newhem   |    2    |       None      |   None   |
+-------------------------+------------+---------+-----------------+----------+

.. note:: \* - default function

"""

from .core import Net

from . import trans
from . import layer
from . import train
from . import error
from . import init
import numpy as np


def newff(minmax, size, transf=None):
    """
    Create multilayer perceptron

    :Parameters:
        minmax: list of list, the outer list is the number of input neurons, 
			inner lists must contain 2 elements: min and max
            Range of input value
        size: the length of list equal to the number of layers except input layer, 
			the element of the list is the neuron number for corresponding layer
            Contains the number of neurons for each layer
        transf: list (default TanSig)
            List of activation function for each layer
    :Returns:
        net: Net
    :Example:
        >>> # create neural net with 2 inputs
		>>> # input range for each input is [-0.5, 0.5]
		>>> # 3 neurons for hidden layer, 1 neuron for output
		>>> # 2 layers including hidden layer and output layer
        >>> net = newff([[-0.5, 0.5], [-0.5, 0.5]], [3, 1])
        >>> net.ci
        2
        >>> net.co
        1
        >>> len(net.layers)
        2

    """

    net_ci = len(minmax)
    net_co = size[-1]

    if transf is None:
        transf = [trans.TanSig()] * len(size)
    assert len(transf) == len(size)

    layers = []
    for i, nn in enumerate(size):
        layer_ci = size[i - 1] if i > 0 else net_ci
        l = layer.Perceptron(layer_ci, nn, transf[i])
        l.initf = init.initnw
        layers.append(l)
    connect = [[i - 1] for i in range(len(layers) + 1)]

    net = Net(minmax, net_co, layers, connect, train.train_bfgs, error.SSE())
    return net


def newp(minmax, cn, transf=trans.HardLim()):
    """
    Create one layer perceptron

    :Parameters:
        minmax: list of list, the outer list is the number of input neurons, 
			inner lists must contain 2 elements: min and max
            Range of input value
        cn: int, number of output neurons
            Number of neurons
        transf: func (default HardLim)
            Activation function
    :Returns:
        net: Net
    :Example:
        >>> # create network with 2 inputs and 10 neurons
        >>> net = newp([[-1, 1], [-1, 1]], 10)

    """

    ci = len(minmax)
    l = layer.Perceptron(ci, cn, transf)
    net = Net(minmax, cn, [l],  [[-1], [0]], train.train_delta, error.SSE())
    return net


def newc(minmax, cn):
    """
    Create competitive layer (Kohonen network)

    :Parameters:
        minmax: list of list, the outer list is the number of input neurons, 
			inner lists must contain 2 elements: min and max
            Range of input value
        cn: int, number of output neurons
            Number of neurons
    :Returns:
        net: Net
    :Example:
        >>> # create network with 2 inputs and 10 neurons
        >>> net = newc([[-1, 1], [-1, 1]], 10)

    """
    ci = len(minmax)
    l = layer.Competitive(ci, cn)
    net = Net(minmax, cn, [l], [[-1], [0]], train.train_cwta, error.SAE())

    return net


def newlvq(minmax, cn0, pc):
    """
    Create a learning vector quantization (LVQ) network

    :Parameters:
        minmax: list of list, the outer list is the number of input neurons, 
			inner lists must contain 2 elements: min and max
            Range of input value
        cn0: int
            Number of neurons in input layer
        pc: list
            List of percent, sum(pc) == 1
    :Returns:
        net: Net
    :Example:
        >>> # create network with 2 inputs,
        >>> # 2 layers and 10 neurons in each layer
        >>> net = newlvq([[-1, 1], [-1, 1]], 10, [0.6, 0.4])

    """
    pc = np.asfarray(pc)
    assert sum(pc) == 1
    ci = len(minmax)
    cn1 = len(pc)
    assert cn0 > cn1

    layer_inp = layer.Competitive(ci, cn0)
    layer_out = layer.Perceptron(cn0, cn1, trans.PureLin())
    layer_out.initf = None
    layer_out.np['b'].fill(0.0)
    layer_out.np['w'].fill(0.0)
    inx = np.floor(cn0 * pc.cumsum())
    for n, i in enumerate(inx):
        st = 0 if n == 0 else inx[n - 1]
        layer_out.np['w'][n][st:i].fill(1.0)
    net = Net(minmax, cn1, [layer_inp, layer_out],
                            [[-1], [0], [1]], train.train_lvq, error.MSE())

    return net


def newelm(minmax, size, transf=None):
    """
    Create a Elman recurrent network

    :Parameters:
        minmax: list of list, the outer list is the number of input neurons, 
			inner lists must contain 2 elements: min and max
            Range of input value
        size: the length of list equal to the number of layers except input layer, 
			the element of the list is the neuron number for corresponding layer
            Contains the number of neurons for each layer
    :Returns:
        net: Net
    :Example:
		>>> # 1 input, input range is [-1, 1], 1 output neuron, 1 layer including output layer
        >>> net = newelm([[-1, 1]], [1], [trans.PureLin()])
        >>> net.layers[0].np['w'][:] = 1 # set weight for all input neurons to 1
        >>> net.layers[0].np['b'][:] = 0 # set bias for all input neurons to 0
        >>> net.sim([[1], [1] ,[1], [3]])
        array([[ 1.],
               [ 2.],
               [ 3.],
               [ 6.]])
    """

    net_ci = len(minmax)
    net_co = size[-1]

    if transf is None:
        transf = [trans.TanSig()] * len(size)
    assert len(transf) == len(size)

    layers = []
    for i, nn in enumerate(size):
        layer_ci = size[i - 1] if i > 0 else net_ci + size[0]
        l = layer.Perceptron(layer_ci, nn, transf[i])
        #l.initf = init.InitRand([-0.1, 0.1], 'wb')
        layers.append(l)
    connect = [[i - 1] for i in range(len(layers) + 1)]
    # recurrent set
    connect[0] = [-1, 0]

    net = Net(minmax, net_co, layers, connect, train.train_gdx, error.MSE())
    return net


def newhop(target, transf=None, max_init=10, delta=0):
    """
    Create a Hopfield recurrent network

    :Parameters:
        target: array like (l x net.co)
            train target patterns
        transf: func (default HardLims)
            Activation function
        max_init: int (default 10)
            Maximum of recurrent iterations
        delta: float (default 0)
            Minimum difference between 2 outputs for stop recurrent cycle
    :Returns:
        net: Net
    :Example:
        >>> net = newhem([[-1, -1, -1], [1, -1, 1]])
        >>> output = net.sim([[-1, 1, -1], [1, -1, 1]])

    """

    target = np.asfarray(target)
    assert target.ndim == 2

    ci = len(target[0])
    if transf is None:
        transf = trans.HardLims()
    l = layer.Reccurent(ci, ci, transf, max_init, delta)
    w = l.np['w']
    b = l.np['b']

    # init weight
    for i in range(ci):
        for j in range(ci):
            if i == j:
                w[i, j] = 0.0
            else:
                w[i, j] = np.sum(target[:, i] * target[:, j]) / ci
        b[i] = 0.0
    l.initf = None

    minmax = transf.out_minmax if hasattr(transf, 'out_minmax') else [-1, 1]

    net = Net([minmax] * ci, ci, [l], [[-1], [0]], None, None)
    return net


def newhem(target, transf=None, max_iter=10, delta=0):
    """
    Create a Hemming recurrent network with 2 layers

    :Parameters:
        target: array like (l x net.co)
            train target patterns
        transf: func (default SatLinPrm(0.1, 0, 10))
            Activation function of input layer
        max_init: int (default 10)
            Maximum of recurrent iterations
        delta: float (default 0)
            Minimum dereference between 2 outputs for stop recurrent cycle
    :Returns:
        net: Net
    :Example:
        >>> net = newhop([[-1, -1, -1], [1, -1, 1]])
        >>> output = net.sim([[-1, 1, -1], [1, -1, 1]])

    """

    target = np.asfarray(target)
    assert target.ndim == 2

    cn = target.shape[0]
    ci = target.shape[1]

    if transf is None:
        transf = trans.SatLinPrm(0.1, 0, 10)
    layer_inp = layer.Perceptron(ci, cn, transf)

    # init input layer
    layer_inp.initf = None
    layer_inp.np['b'][:] = float(ci) / 2
    for i, tar in enumerate(target):
        layer_inp.np['w'][i][:] = tar / 2

    layer_out = layer.Reccurent(cn, cn, trans.SatLinPrm(1, 0, 1e6), max_iter, delta)
    # init output layer
    layer_out.initf = None
    layer_out.np['b'][:] = 0
    eps = - 1.0 / cn
    for i in range(cn):
        layer_out.np['w'][i][:] = [eps] * cn
        layer_out.np['w'][i][i] = 1
    # create network
    minmax = [[-1, 1]] * ci
    layers = [layer_inp, layer_out]
    connect = [[-1], [0], [1]]
    net = Net(minmax, cn, layers, connect, None, None)
    return net
