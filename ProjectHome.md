# NeuroLab #
**Neurolab** is a simple and powerful Neural Network Library for Python.
Contains based neural networks, train algorithms and flexible framework to create and explore other networks


## Features ##

  * Pure python + numpy
  * API like Neural Network Toolbox (NNT) from MATLAB
  * Interface to use train algorithms form scipy.optimize
  * Flexible network configurations and learning algorithms. You may change: train, error, initializetion and activation functions
  * Variety of supported types of Artificial Neural Network and learning algorithms
  * Python 3 support
<a href='Hidden comment: 
==Donate==
You can support the development of the project [https://www.paypal.com/cgi-bin/webscr?hosted_button_id=HRF5T66LM4L7G&cmd=_s-xclick https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif]
'></a>

## Example ##
```
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
```

## Install ##

Install **neurolab** using setuptools/distribute:

```
    easy_install neurolab
```

Or pip:

```
    pip install neurolab
```

Or, if you don't have setuptools/distribute installed,
use the download [link](http://pypi.python.org/pypi/neurolab) at right to download the source package,
and install it in the normal fashion: Unzip the source package,
cd to the new directory, and:

```
    python setup.py install
```

## Support neural networks types ##

  * ### Single layer perceptron ###
    * create function: [neurolab.net.newp()](http://packages.python.org/neurolab/lib.html#neurolab.net.newp)
    * example of use: [newp](http://packages.python.org/neurolab/ex_newp.html)
    * default train function: [neurolab.train.train\_delta()](http://packages.python.org/neurolab/lib.html#neurolab.train.train_delta)
    * support train functions: train\_gd, train\_gda, train\_gdm, train\_gdx, train\_rprop, train\_bfgs, train\_cg

  * ### Multilayer feed forward perceptron ###
    * create function: [neurolab.net.newff()](http://packages.python.org/neurolab/lib.html#neurolab.net.newff)
    * example of use: [newff](http://packages.python.org/neurolab/ex_newff.html)
    * default train function: [neurolab.train.train\_gdx()](http://packages.python.org/neurolab/lib.html#neurolab.train.train_gdx)
    * support train functions: train\_gd, train\_gda, train\_gdm, train\_rprop, train\_bfgs, train\_cg

  * ### Competing layer (Kohonen Layer) ###
    * create function: [neurolab.net.newc()](http://packages.python.org/neurolab/lib.html#neurolab.net.newc)
    * example of use: [newc](http://packages.python.org/neurolab/ex_newc.html)
    * default train function: [neurolab.train.train\_cwta()](http://packages.python.org/neurolab/lib.html#neurolab.train.train_cwta)
    * support train functions: train\_wta

  * ### Learning Vector Quantization (LVQ) ###
    * create function: [neurolab.net.newlvq()](http://packages.python.org/neurolab/lib.html#neurolab.net.newlvq)
    * example of use: [newlvq](http://packages.python.org/neurolab/ex_newlvq.html)
    * default train function: [neurolab.train.train\_lvq()](http://packages.python.org/neurolab/lib.html#neurolab.train.train_lvq)

  * ### Elman Recurrent network ###
    * create function: [neurolab.net.newelm()](http://packages.python.org/neurolab/lib.html#neurolab.net.newelm)
    * example of use: [newelm](http://packages.python.org/neurolab/ex_newelm.html)
    * default train function: [neurolab.train.train\_gdx()](http://packages.python.org/neurolab/lib.html#neurolab.train.train_gdx)
    * support train functions: train\_gd, train\_gda, train\_gdm, train\_rprop, train\_bfgs, train\_cg
  * ### Hopfield Recurrent network ###
    * create function: [neurolab.net.newhop()](http://packages.python.org/neurolab/lib.html#neurolab.net.newhop)
    * example of use: [newhop](http://packages.python.org/neurolab/ex_newhop.html)
  * ### Hemming Recurrent network ###
    * create function: [neurolab.net.newhem()](http://packages.python.org/neurolab/lib.html#neurolab.net.newhem)
    * example of use: [newhem](http://packages.python.org/neurolab/ex_newhem.html)