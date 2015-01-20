# -*- coding: utf-8 -*-
"""
Train algorithm based on Winner Take All - rule

"""

from neurolab.core import Train
import neurolab.tool as tool
import numpy as np

class TrainWTA(Train):
    """ 
    Winner Take All algorithm
    
    :Support networks:
        newc (Kohonen layer)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """
       
    def __init__(self, net, input, lr=0.01):
        # Init network!
        self.lr = lr
        for w in net.layers[0].np['w']:
            w[:] = input[np.random.randint(0, len(input))]

    def error(self, net, input):
        layer = net.layers[0]
        winner_output = np.zeros_like(input)
        output = net.sim(input)
        winners = np.argmax(output, axis=1)
        
        return net.errorf(layer.np['w'][winners], input)
    
    def learn(self, net, input):
        layer = net.layers[0]

        for inp in input:
            out = net.step(inp)
            winner = np.argmax(out)
            d = layer.last_dist
            layer.np['w'][winner] += self.lr * d[winner] * (inp - layer.np['w'][winner])
        
        return None


class TrainCWTA(TrainWTA):
    """ 
    Conscience Winner Take All algorithm
    
    :Support networks:
        newc (Kohonen layer)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """
    
    def learn(self, net, input):
        layer = net.layers[0]

        for inp in input:
            out = net.step(inp)
            winner = np.argmax(out)
            d = layer.last_dist #TODO:^^_^^
            layer.np['conscience'][winner] += 1
            layer.np['w'][winner] += self.lr * d[winner] * (inp - layer.np['w'][winner])

        layer.np['conscience'].fill(1.0)
        return None