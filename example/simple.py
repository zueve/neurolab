# -*- coding: utf-8 -*-
""" 
Simple example of use neurolab
==============================

"""

import numpy as np
import neurolab as nl
# Create train samples
input = np.random.uniform(-0.5, 0.5, (10, 2))
target = (input[:, 0] + input[:, 1]).reshape(10, 1)
# Create network with 2 inputs, 5 neurons in input layer and 1 in output layer
net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
# Train process
err = net.train(input, target, show=15)
# Test
net.sim([[0.2, 0.1]]) # 0.2 + 0.1