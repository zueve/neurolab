*************************
Modified network property
*************************
::

	>>> import neurolab as nl
	>>> # Create network
	>>> net = nl.net.newff([[-1, 1]], [5, 1])
	>>> # Default train function (train_gdx)
	>>> print net.trainf
	Trainer(TrainGDX)
	>>> # Change train function
	>>> net.trainf = nl.train.train_bfgs
	>>> # Change init function
	>>> net.initf = nl.init.InitRand([-2., 2.], ['w', 'b'])
	>>> # new inicialized
	>>> net.init()
	>>> # Change error function
	>>> net.errorf = nl.error.MSE()
	>>> # Change weight of input layer
	>>> net.layers[0].np['w'][:] = 0.0
	>>> net.layers[0].np['w']
	>>> array([[ 0.],
		   [ 0.],
		   [ 0.],
		   [ 0.],
		   [ 0.]])
	>>> # Change bias of input layer
	>>> net.layers[0].np['b'][:] = 1.0
	>>> net.layers[0].np['b']
	array([ 1.,  1.,  1.,  1.,  1.])
	>>> # Save network in file
	>>> net.save('test.net')
	>>> # Load network
	>>> net = nl.load('test.net')