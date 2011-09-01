# -*- coding: utf-8 -*-
"""
Train algorithm based on spipy.optimize

"""

from neurolab.core import Train
import neurolab.tool as tool

class TrainBFGS(Train):
    """
    Broyden–Fletcher–Goldfarb–Shanno (BFGS) method
    Using scipy.optimize.fmin_bfgs
    
    :Support networks:
        newff (multy-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """
    
    def __init__(self, net, input, target, **kwargs):
        self.kwargs = kwargs
        self.x = tool.np_get_ref(net)
        
    def __call__(self, net, input, target):
        def grad(x):
            #print 'grad'
            self.x[:] = x
            gr = tool.ff_grad(net, input, target)[1]
            return gr
        
        def fcn(x):
            #print 'fcn'
            self.x[:] = x
            err = self.error(net, input, target)
            self.lerr = err
            return err
        
        def step(x):
            #print 'step'
            self.x[:] = x
            err = self.error(net, input, target)
            self.epochf(err, net, input, target)
        self.opt(fcn, step, grad)
        
    def opt(self, fcn, step, grad):
        from scipy.optimize import fmin_bfgs
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['maxiter'] = self.epochs
        x = fmin_bfgs(fcn, self.x.copy(), fprime=grad, callback=step, **self.kwargs)
        self.x[:] = x
        return None


class TrainCG(TrainBFGS):
    """
    Conjugate gradient algorithm
    Using scipy.optimize.fmin_cg
    
    :Support networks:
        newff (multy-layers perceptron)  
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
    
    """

    def opt(self, fcn, step, grad):
        from scipy.optimize import fmin_cg 
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        x = fmin_cg(fcn, self.x.copy(), fprime=grad, callback=step, **self.kwargs)
        self.x[:] = x
        return None