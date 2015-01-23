# -*- coding: utf-8 -*-
"""
Neurolab is a simple and powerful Neural Network Library for Python.
Contains based neural networks, train algorithms and flexible framework
to create and explore other neural network types.


:Features:

    - Pure python + numpy
    - API like Neural Network Toolbox (NNT) from MATLAB
    - Interface to use train algorithms form scipy.optimize
    - Flexible network configurations and learning algorithms. You may change: train, error, initialization and activation functions
    - Unlimited number of neural layers and number of neurons in layers
    - Variety of supported types of Artificial Neural Network and learning algorithms

:Example:

	>>> import numpy as np
	>>> import neurolab as nl
	>>> # Create train samples
	>>> input = np.random.uniform(-0.5, 0.5, (10, 2))
	>>> target = (input[:, 0] + input[:, 1]).reshape(10, 1)
	>>> # Create network with 2 inputs, 5 neurons in input layer and 1 in output layer
	>>> net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
	>>> # Train process
	>>> err = net.train(input, target, show=15)
	Epoch: 15; Error: 0.150308402918;
	Epoch: 30; Error: 0.072265865089;
	Epoch: 45; Error: 0.016931355131;
	The goal of learning is reached
	>>> # Test
	>>> net.sim([[0.2, 0.1]]) # 0.2 + 0.1
	array([[ 0.28757596]])

:Links:

    - `Home Page <http://code.google.com/p/neurolab/>`_
    - `PyPI Page <http://pypi.python.org/pypi/neurolab>`_
    - `Documentation <http://packages.python.org/neurolab/>`_
    - `Examples <http://packages.python.org/neurolab/example.html>`_


"""


__version__ = '0.3.5'

# Development Status :: 1 - Planning, 2 - Pre-Alpha, 3 - Alpha,
#                       4 - Beta, 5 - Production/Stable
__status__ = '3 - Alpha'

from .tool import load
from . import net
