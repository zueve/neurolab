# -*- coding: utf-8 -*-
"""
Neurolab is a simple and powerful Neural Network Library for Python.
Contains based neural networks, train algorithms and flexible framework 
to create and explore other networks.


:Features:

    - Pure python + numpy
    - API like Neural Network Toolbox (NNT) from MATLAB
    - Interface to use train algorithms form scipy.optimize
    - Flexible network configurations and learning algorithms. You may change: train, error, initializetion and activation functions
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


:Now support neural networks types:

	**Single layer perceptron**
		- create function: `neurolab.net.newp() <http://packages.python.org/neurolab/lib.html#neurolab.net.newp>`_
		- example of use: `newp <http://packages.python.org/neurolab/ex_newp.html>`_
		- default train function: `neurolab.train.train_delta() <http://packages.python.org/neurolab/lib.html#neurolab.train.train_delta>`_
		- support train functions: train_gd, train_gda, train_gdm, train_gdx, train_rprop, train_bfgs, train_cg
	
	**Multilayer feed forward perceptron**
		- create function: `neurolab.net.newff() <http://packages.python.org/neurolab/lib.html#neurolab.net.newff>`_
		- example of use: `newff <http://packages.python.org/neurolab/ex_newff.html>`_
		- default train function: `neurolab.train.train_gdx() <http://packages.python.org/neurolab/lib.html#neurolab.train.train_gdx>`_
		- support train functions: train_gd, train_gda, train_gdm, train_rprop, train_bfgs, train_cg
		
	**Competing layer (Kohonen Layer)**
		- create function: `neurolab.net.newc() <http://packages.python.org/neurolab/lib.html#neurolab.net.newc>`_
		- example of use: `newc <http://packages.python.org/neurolab/ex_newc.html>`_
		- default train function: `neurolab.train.train_cwta() <http://packages.python.org/neurolab/lib.html#neurolab.train.train_cwta>`_
		- support train functions: train_wta
		
	**Learning Vector Quantization (LVQ)**
		- create function: `neurolab.net.newlvq() <http://packages.python.org/neurolab/lib.html#neurolab.net.newlvq>`_
		- example of use: `newlvq <http://packages.python.org/neurolab/ex_newlvq.html>`_
		- default train function: `neurolab.train.train_lvq() <http://packages.python.org/neurolab/lib.html#neurolab.train.train_lvq>`_

"""
import layer
import net
from tool import load


__version__ = '0.1.1'

# Development Status :: 1 - Planning, 2 - Pre-Alpha, 3 - Alpha, 
#                       4 - Beta, 5 - Production/Stable
__status__ = '3 - Alpha'
