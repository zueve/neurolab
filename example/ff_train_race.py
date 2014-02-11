# -*- coding: utf-8 -*-
"""
Testing train algoritms
=======================

"""

import neurolab as nl
import numpy as np
import pylab as pl
import time

# Approximation function: 1/2 * sin(x)
x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

size = len(x)

input = x.reshape(size, 1)
target = y.reshape(size, 1)

# Create network with 2 layers and random initialized
net = nl.net.newff([[-7, 7]], [5, 1])

# Train algorinms
trains = [nl.train.train_gd, 
          nl.train.train_gdm, 
          nl.train.train_gda, 
          nl.train.train_gdx,
          nl.train.train_rprop, 
          nl.train.train_cg,
          #nl.train.train_ncg,
          nl.train.train_bfgs,
          ]
# Test process
errors, times = [], []
for train in trains:
    net.trainf = train
    cnet = net.copy()
    # cold start
    cnet.train(input, target, epochs=1, show=0, goal=0)
    cnet = net.copy()
    st = time.clock()
    # hot start
    e = cnet.train(input, target, epochs=100, show=0, goal=0)
    times.append(time.clock() - st)
    errors.append(e)
    
# Plot result
lables = [t._train_class.__name__[5:] for t in trains]
ind = np.arange(len(errors))
width = 0.8
pl.subplot(211)
pl.bar(ind, [e[-1] for e in errors], width)
pl.ylabel('Result error')
pl.xticks(ind + width/2, lables)

pl.subplot(212)
pl.bar(ind, times, width)
pl.ylabel('Time work')
pl.xticks(ind + width/2, lables)

pl.show()