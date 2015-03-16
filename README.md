# Neurolab #

**Neurolab** is a simple and powerful Neural Network Library for Python.
Contains based neural networks, train algorithms and flexible framework 
to create and explore other neural network types.


## Features ##

- Pure python + numpy
- API like Neural Network Toolbox (NNT) from **MATLAB**
- Interface to use train algorithms form **scipy.optimize**
- Flexible network configurations and learning algorithms. You may change: train, error, initializetion and activation functions
- Unlimited number of neural layers and number of neurons in layers
- Variety of supported types of Artificial Neural Network and learning algorithms

## Example ##

```
    >>> import numpy as np
    >>> import neurolab as nl
    >>> # Create train samples
    >>> input = np.random.uniform(-0.5, 0.5, (10, 2))
    >>> target = (input[:, 0] + input[:, 1]).reshape(10, 1)
    >>> # Create network with 2 inputs, 5 neurons in input layer
    >>> # And 1 in output layer
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
```

## Links ##

- Home Page: <http://code.google.com/p/neurolab/>
- PyPI Page: <http://pypi.python.org/pypi/neurolab>
- Documentation: <http://packages.python.org/neurolab/>
- Examples: <http://packages.python.org/neurolab/example.html>

## Install ##

Install *neurolab* using pip:
    
```
    $> pip install neurolab
```
    
Or, if you don't have setuptools/distribute installed, 
use the download [link](https://github.com/zueve/neurolab/releases) 
at right to download the source package, and install it in the normal fashion. Ungzip and untar the source package, cd to the new directory, and:

```
    $> python setup.py install
```