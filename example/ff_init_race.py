# coding: utf-8
"""
Testing init algoritms
=======================

"""

import neurolab as nl
import numpy as np
import pylab as pl

TRAIN_CYCLES = 100
TRAIN_EPOCHS = 100

# Approximation function: 1/2 * sin(x)
x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

input = x.reshape(len(x), 1)
target = y.reshape(len(x), 1)

# Create network with 2 layers and random initialized
net = nl.net.newff([[-7, 7]], [10, 1])
net.trainf = nl.train.train_gdx
# Init algorinms
inits = {'reg': nl.init.initwb_reg,
         'rand [-1, 1]': nl.init.InitRand([-1.0, 1.0], 'wb'),
         'rand [-0.5,0.5]': nl.init.InitRand([-.5, .5], 'wb'),
         'initnw': nl.init.initnw
         }

# Test process
errors = []
labels = []
for name, init in inits.items():
    error = np.zeros(TRAIN_EPOCHS)
    for l in net.layers:
        l.initf = init
    for i in range(TRAIN_CYCLES):
        net.init()
        err = net.train(input, target, epochs=TRAIN_EPOCHS, show=0, goal=0)
        error += err
    labels.append(name)
    errors.append(error / TRAIN_CYCLES)
errors = np.array(errors)
# Plot result
width = 0.8
ind = np.arange(len(inits))

pl.subplot(311)
pl.plot(errors.T)
pl.legend(labels)

pl.subplot(312)
pl.bar(ind, errors[:, -1], width)
pl.ylabel('AVG of last error')
pl.xticks(ind + width/2, labels)
pl.subplot(313)
pl.bar(ind, errors.mean(axis=1), width)
pl.ylabel('AVG error of all train proces')
pl.xticks(ind + width/2, labels)
pl.show()
