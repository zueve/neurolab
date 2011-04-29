# -*- coding: utf-8 -*-
"""
The module contains the basic network architectures



+-------------------------+------------+---------+-----------------+----------+
|      Network Type       |  Function  | Count of|Support train fcn| Error fcn|
|                         |            | layers  |                 |          |
+=========================+============+=========+=================+==========+
| Single-layer perceptron |    newp    |    1    |   train_delta   |   SSE    |
+-------------------------+------------+---------+-----------------+----------+
| Multi-layer perceptron  |   newff    |  more 1 |   train_gd,     |   SSE    |
|                         |            |         |   train_gdm,    |          |
|                         |            |         |   train_gda,    |          |
|                         |            |         |   train_gdx*,   |          |
|                         |            |         |   train_rprop,  |          |
|                         |            |         |   train_bfgs,   |          |
|                         |            |         |   train_cg      |          |
+-------------------------+------------+---------+-----------------+----------+
|    Competitive layer    |    newc    |    1    |   train_wta,    |   SAE    |
|                         |            |         |   train_cwta*   |          |
+-------------------------+------------+---------+-----------------+----------+
|           LVQ           |   newlvq   |    2    |   train_lvq     |   MSE    |
+-------------------------+------------+---------+-----------------+----------+

.. note:: \* - default function

"""

from core import Net
import trans
import layer
import train
import error
import numpy as np


def newff(minmax, size, transf=None):
    """
    Create multilayer perceptron

    :Parameters:
        minmax: list ci x 2
            Range of input value
        size: list of length equal to the number of layers
            Contains the number of neurons for each layer
        transf: list (default TanSig)
            List of activation function for each layer
    :Returns:
        net: Net
    :Example:
        >>> # create neural net with 2 inputs, 1 output and 2 layers
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
        layers.append(l)
    connect = [[i - 1] for i in range(len(layers) + 1)]

    net = Net(minmax, net_co, layers, connect, train.train_gdx, error.SSE())
    return net


def newp(minmax, cn, transf=trans.HardLim()):
    """
    Create one layer perceptron

    :Parameters:
        minmax: list ci x 2
            Range of input value
        cn: int
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
        minmax: list ci x 2
            Range of input value
        cn: int
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
        minmax: list ci x 2
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
