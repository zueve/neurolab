# coding: utf-8
""" Setup file for neurolab package """

from distutils.core import setup
import os

def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ''

version = '0.3.5'
status = '5 - Production/Stable'

doc = """
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


setup(name='neurolab',
        version=version,
        description='Simple and powerfull neural network library for python',
        long_description = doc,
        author='Zuev Evgenij',
        author_email='zueves@gmail.com',
        url='http://neurolab.googlecode.com',
        packages=['neurolab', 'neurolab.train'],
        scripts=[],

        classifiers=(
            'Development Status :: ' + status,
            'Environment :: Console',
            'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
			'Programming Language :: Python :: 2',
			'Programming Language :: Python :: 3',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ),
        license="LGPL-3",
        platforms = ["Any"],
        keywords="neural network, neural networks, neural nets, backpropagation, python, matlab, numpy, machine learning"
    )
