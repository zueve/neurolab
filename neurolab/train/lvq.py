# -*- coding: utf-8 -*-
"""
Train algorithms for LVQ networks

"""

from neurolab.core import Train
import neurolab.tool as tool
import numpy as np


class TrainLVQ(Train):
    """
    LVQ1 train function
    
    :Support networks:
        newlvq
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
    
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=True):
        self.adapt = adapt
        self.lr = lr
    
    def __call__(self, net, input, target):
        layer = net.layers[0]
        if self.adapt:
            while True:
                self.epochf(None, net, input, target)
                
                for inp, tar in zip(input, target):
                    out = net.step(inp)
                    err = tar - out
                    win = np.argmax(layer.out)
                    if np.max(err) == 0.0:
                        layer.np['w'][win] += self.lr * (inp - layer.np['w'][win])
                    else:
                        layer.np['w'][win] -= self.lr * (inp - layer.np['w'][win])
        else:
            while True:
                output = []
                winners = []
                for inp, tar in zip(input, target):
                    out = net.step(inp)
                    output.append(out)
                    winners.append(np.argmax(layer.out))
                
                e = self.error(net, input, target, output)
                self.epochf(e, net, input, target)
                
                error = target - output
                sign = np.sign((np.max(error, axis=1) == 0) - 0.5)
                layer.np['w'][winners] += self.lr * (input - layer.np['w'][winners])
        return None