
************
Introduction
************

NeuroLab - a library of base nueral networks algorithms with flexible network configurations and learning algorithms.
To simplify the using of the library, interface is similar to the package of Neural Network Toolbox (NNT) of MATLAB (c).
The library is based on the package numpy (http://numpy.scipy.org), some learning algorithms are used scipy.optymize (http://scipy.org).

:Create network:
	Created two-layer network(3-1) with 2-inputs and one output.
	Input layer contains 3 neurons, the output 1 neuron.
	Input range: 0.0, 0.5::
	
	>>> import neurolab as nl
	>>> # create feed forward multilayer perceptron
	>>> net = nl.net.newff([[0, 0.5], [0, 0.5]], [3, 1])


:Train:
	>>> # Create learning samples
	>>> input = [[0.1, 0.1], 
	...          [0.1, 0.2], 
	...          [0.1, 0.3], 
	...          [0.1, 0.4], 
	...          [0.2, 0.2], 
	...          [0.2, 0.3], 
	...          [0.2, 0.4], 
	...          [0.3, 0.3], 
	...          [0.3, 0.4], 
	...          [0.4, 0.4]]
	>>> target = [[i[0] + i[1]] for i in input]
	>>> # Train
	>>> error = net.train(input, target, epochs=500, goal=0.1)

:Train error:
	>>> print "Finish error:", error[-1]
	Finish error: 0.125232586274

:Simulate:
	>>> net.sim([[0.1, 0.5], [0.3, 0.1]])
	array([[ 0.59650825],
           [ 0.41686071]])

:Network Info:
	>>> # Number of network inputs:
	>>> net.ci
	2
	>>> # Number of network outputs:
	>>> net.co
	1
	>>> # Number of network layers:
	>>> len(net.layers)
	2
	>>> # Weight of first neuron of input layer (net.layers[0])
	>>> net.layers[0].np['w'][1]
	array([-0.67211163, -0.87277918])
	>>> 
	>>> # Bias output layer:
	>>> net.layers[-1].np['b']
	array([-0.69717423])
	>>> # Train params
	>>> net.trainf.defaults
	{'epochs': 500, 
	 'goal': 0.01, 
	 'show': 100,
	 'Train': <class 'neurolab.train.TrainGDX'>, 
	 'TrainParams': {'adapt': False, 
					 'lr': 0.01, 
					 'max_perf_inc': 1.04, 
					 'mc': 0.90000000000000002, 
					 'lr_dec': 0.69999999999999996, 
					 'lr_inc': 1.05}
	 }

:Save/Load:
	>>> net.save('sum.net')
	>>> newnet = nl.load('sum.net')

:Change train function:
	>>> net.trainf = nl.train.train_cg
	>>> # Change error function:
	>>> net.errorf = nl.error.SAE()

:Change transfer function on output layer:
	>>> net.layers[-1].transf = nl.trans.HardLim()