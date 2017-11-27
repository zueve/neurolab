# -*- coding: utf-8 -*-
"""
Example of use probabilistic network
=====================================

"""

import neurolab as nl
import numpy as np

# Create train samples
input = np.array([[-2, -2], [-1, 0], [-0.5, -1], [-0.5, 1], [0, 0], [0, 2], [0.5, -1],
                                                            [0.5, 1], [1, 0], [2, -2]])
target = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1],
                                                        [0, 1], [1, 0], [1, 0]])

# Create and train network with 2 layers:10 neurons in input layer(RBN)
# and 2 neurons in output layer(Competitive)
# std = 2
net = nl.net.newpnn(nl.tool.minmax(input), input, target, 2)

# Plot result
import pylab as pl
xx, yy = np.meshgrid(np.arange(-3, 3.4, 0.1), np.arange(-3, 3.4, 0.1))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
i = np.concatenate((xx, yy), axis=1)
o = net.sim(i)
grid1 = i[o[:, 0]>0]
grid2 = i[o[:, 1]>0]

class1 = input[target[:, 0]>0]
class2 = input[target[:, 1]>0]

pl.plot(class1[:,0], class1[:,1], 'ro', class2[:,0], class2[:,1], 'go')
pl.plot(grid1[:,0], grid1[:,1], 'rs', grid2[:,0], grid2[:,1], 'gs', alpha=0.2)
pl.axis([-3.2, 3.2, -3, 3])
pl.xlabel('Input[:, 0]')
pl.ylabel('Input[:, 1]')
pl.legend(['class 1', 'class 2', 'detected class 1', 'detected class 2'])
pl.show()