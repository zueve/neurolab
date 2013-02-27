***********************************
Hemming Recurrent network (newhem)
***********************************

Use  :py:func:`neurolab.net.newhem`


.. literalinclude:: ../../example/newhem.py

:Result:

::

    Test on train samples (must be [0, 1, 2, 3, 4])
	[0 1 2 3 4]
	Outputs on recurent cycle:
	[[ 0.       0.24     0.48     0.       0.     ]
	 [ 0.       0.144    0.432    0.       0.     ]
	 [ 0.       0.0576   0.4032   0.       0.     ]
	 [ 0.       0.       0.39168  0.       0.     ]]
	Outputs on test sample:
	[[ 0.          0.          0.39168     0.          0.        ]
	 [ 0.          0.          0.          0.          0.39168   ]
	 [ 0.07516193  0.          0.          0.          0.07516193]]

