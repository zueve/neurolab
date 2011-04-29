# -*- coding: utf-8 -*-
"""
Train algorithm based on Delta - rule

"""

from neurolab.core import Train
import neurolab.tool as tool


class TrainDelta(Train):
    """ 
    Train with Delta rule
    
    :Support networks:
        newp (one-layer perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        lr: float (default 0.01)
            learning rate
    
    """
    
    def __init__(self, net, input, target, lr=0.01):
        self.lr = lr
        
    def __call__(self, net, input, target):
        layer = net.layers[0]
        while True:
            e = self.error(net, input, target)
            self.epochf(e, net, input, target)
            for inp, tar in zip(input, target):
                out = net.step(inp)
                err = tar - out
                err.shape =  err.size, 1
                inp.shape = 1, inp.size
                layer.np['w'] += self.lr * err * inp
                err.shape =  err.size
                layer.np['b'] += self.lr * err
        return None
