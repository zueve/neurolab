# -*- coding: utf-8 -*-
import numpy as np
import neurolab.tool as tool
from neurolab.core import TrainStop



        
        
class TrainWTA(Train):
    """ 
    Winner Take All algorithm
    
    :Support networks:
        newc (kohonen layer)
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
        e =  layer.np['w'][winners] - input
        
        return net.errorf(e)
    
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
    Conscience Winner Take All algoritm
    
    :Support networks:
        newc (cohonen layer)
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



        

class TrainDelta(Train):
    """ 
    Train with Delta rule 
    
    :Support networks:
        newp (one-layers perceptron)
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
