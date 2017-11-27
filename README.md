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

## Support neural networks types ##

- **Single layer perceptron**
    - create function: [neurolab.net.newp()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newp)
    - example of use: [newp](https://pythonhosted.org/neurolab/ex_newp.html)
    - default train function: [neurolab.train.train_delta()](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_delta)
    - support train functions: [train_gd](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gd),
                               [train_gda](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gda),
                               [train_gdm](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gdm),
                               [train_gdx](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gdx),
                               [train_rprop](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_rprop),
                               [train_bfgs](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_bfgs),
                               [train_cg](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_cg)
- **Multilayer feed forward perceptron**
    - create function: [neurolab.net.newff()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newff)
    - example of use: [newff](https://pythonhosted.org/neurolab/ex_newff.html)
    - default train function: [neurolab.train.train_bfgs()](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_bfgs)
    - support train functions: [train_gd](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gd),
                               [train_gda](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gda),
                               [train_gdm](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gdm),
                               [train_rprop](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_rprop),
                               [train_bfgs](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_bfgs),
                               [train_cg](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_cg)
- **Competing layer (Kohonen Layer)**
    - create function: [neurolab.net.newc()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newc)
    - example of use: [newc](https://pythonhosted.org/neurolab/ex_newc.html)
    - default train function: [neurolab.train.train_cwta()](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_cwta)
    - support train functions: [train_wta](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_wta)
- **Learning Vector Quantization (LVQ)**
    - create function: [neurolab.net.newlvq()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newlvq)
    - example of use: [newlvq](https://pythonhosted.org/neurolab/ex_newlvq.html)
    - default train function: [neurolab.train.train_lvq()](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_lvq)
- **Elman Recurrent network**
    - create function: [neurolab.net.newelm()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newelm)
    - example of use: [newelm](https://pythonhosted.org/neurolab/ex_newelm.html)
    - default train function: [neurolab.train.train_gdx()](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gdx)
    - support train functions: [train_gd](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gd),
                               [train_gda](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gda),
                               [train_gdm](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_gdm),
                               [train_rprop](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_rprop),
                               [train_bfgs](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_bfgs),
                               [train_cg](https://pythonhosted.org/neurolab/lib.html#neurolab.train.train_cg)
- **Hopfield Recurrent network**
    - create function: [neurolab.net.newhop()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newhop)
    - example of use: [newhop](https://pythonhosted.org/neurolab/ex_newhop.html)
- **Hemming Recurrent network**
    - create function: [neurolab.net.newhem()](https://pythonhosted.org/neurolab/lib.html#neurolab.net.newhem)
    - example of use: [newhem](https://pythonhosted.org/neurolab/ex_newhem.html)
- **Generalized Regression network**
    - create function: [neurolab.net.newgrnn()]
    - example of use: [newgrnn]
- **Probabilistic network**
    - create function: [neurolab.net.newpnn()]
    - example of use: [newpnn]