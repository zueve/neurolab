# -*- coding: utf-8 -*-
""" 
Example of use singe layer perceptron(newp)
===========================================

Train with Delta rule

"""

import neurolab as nl

# Logical &
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

# Create net with 2 inputs and 1 neuron
net = nl.net.newp([[0, 1],[0, 1]], 1)

# train with delta rule
# see net.trainf
error = net.train(input, target, epochs=100, show=10, lr=0.1)

# Plot results
import pylab as pl
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()
