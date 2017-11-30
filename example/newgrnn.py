# -*- coding: utf-8 -*-
"""
Example of use generalized regression neural network
=====================================

Task: Approximation function: 1/2 * exp(x)

"""

import neurolab as nl
import numpy as np

# Create train samples
x = np.linspace(0, 4, 15)
y = np.exp(x) * 0.5

size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)

# Create and train network with 2 layers: radial basis layer and linear layer; std = 2
net = nl.net.newgrnn(nl.tool.minmax(inp), inp, tar, 2)

# Simulate network
out = net.sim(inp)

# Plot result
import pylab as pl

x2 = np.linspace(-0.2,3.8,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

y3 = out.reshape(size)

pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['net output', 'train target', 'net output for train target'])
pl.show()
