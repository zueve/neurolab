# -*- coding: utf-8 -*-
"""
Train algorithm based  gradients algorithms

"""
import numpy as np
from neurolab.core import Train
import neurolab.tool as tool

class TrainGD(Train):
    """
    Gradient descent backpropogation
    
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
        rr float (defaults 0.0)
            Regularization ratio
            Must be between {0, 1}
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=False, rr=0.0):
        self.adapt = adapt
        self.lr = lr
        self.rr = rr
        
    def __call__(self, net, input, target):

        if not self.adapt:
            while True:
                g, output = self.calc(net, input, target)
                # regularization grad
                if self.rr > 0:
                    g = tool.reg_grad(g, net, self.rr)
                e = self.error(net, input, target, output)
                # regularization error
                if self.rr:
                    e = tool.reg_error(e, net, self.rr)
                self.epochf(e, net, input, target)
                self.learn(net, g)
        else:
            while True:
                for i in range(input.shape[0]):
                    g = self.calc(net, [input[i]], [target[i]])[0]
                    # regularization grad
                    if self.rr > 0:
                        g = tool.reg_grad(g, net, self.rr)
                    self.learn(net, g)
                e = self.error(net, input, target)
                # regularization error
                if self.rr:
                    e = reg_error(e, self.rr, net)
                self.epochf(e, net, input, target)
        return None
            
    def calc(self, net, input, target):
        g1, g2, output = tool.ff_grad(net, input, target)
        return g1, output
    
    def learn(self, net, grad):
        for ln, layer in enumerate(net.layers):
            layer.np['w'] -= self.lr * grad[ln]['w']
            layer.np['b'] -= self.lr * grad[ln]['b']
        return None
        

class TrainGD2(TrainGD):
    """
    Gradient descent backpropagation
    (another realization of TrainGD)
    
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
        
    """
    
    def __init__(self, net, input, target, lr=0.01, adapt=False):
        self.adapt = adapt
        self.lr = lr
        self.x = tool.np_get_ref(net)
        # Regularization not suppotr
        self.rr = 0 
    
    def calc(self, net, input, target):
        g1, g2, output = tool.ff_grad(net, input, target)
        return g2, output
    
    def learn(self, net, grad):
        self.x -= self.lr * grad

        
class TrainGDM(TrainGD):
    """
    Gradient descent with momentum backpropagation
    
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.01)
            learning rate
        adapt bool (default False)
            type of learning
        mc: float (default 0.9)
            Momentum constant
        rr float (defaults 0.0)
            Regularization ratio
            Must be between {0, 1}
    
    """

    def __init__(self, net, input, target, lr=0.01, adapt=False, mc=0.9, rr=.0):
        super(TrainGDM, self).__init__(net, input, target, lr, adapt, rr)
        self.mc = mc
        self.dw = [0] * len(net.layers)
        self.db = [0] * len(net.layers)
    
    def learn(self, net, grad):
        #print 'GDM.learn'
        mc = self.mc
        lr = self.lr
        for ln, layer in enumerate(net.layers):
            self.dw[ln] = mc * self.dw[ln] + ((1 - mc) * lr) * grad[ln]['w'] 
            self.db[ln] = mc * self.db[ln] + ((1 - mc) * lr) * grad[ln]['b']
            layer.np['w'] -= self.dw[ln]
            layer.np['b'] -= self.db[ln]
        return None

class TrainGDA(TrainGD):
    """
    Gradient descent with adaptive learning rate
    
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.01)
            learning rate
        adapt: bool (default False)
            type of learning
        lr_inc: float (> 1, default 1.05)
            Ratio to increase learning rate
        lr_dec: float (< 1, default 0.7)
            Ratio to decrease learning rate
        max_perf_inc:float (> 1, default 1.04)
            Maximum performance increase
        rr float (defaults 0.0)
            Regularization ratio
            Must be between {0, 1}
    
    """
    def __init__(self, net, input, target, lr=0.01, adapt=False, 
                                                    lr_inc=1.05, lr_dec=0.7, 
                                                    max_perf_inc=1.04, rr=.0):
        super(TrainGDA, self).__init__(net, input, target, lr, adapt, rr)
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.max_perf_inc = max_perf_inc
        self.err = []

    def learn(self, net, grad):
        #print 'GDA.learn'
        if len(self.err) > 1:
            f = self.err[-1] / self.err[-2]
            if f > self.max_perf_inc:
                self.lr *= self.lr_dec
            elif f < 1:
                self.lr *= self.lr_inc
        super(TrainGDA, self).learn(net, grad)
        return None
    
    def error(self, *args, **kwargs):
        e = super(TrainGDA, self).error(*args, **kwargs)
        self.err.append(e)
        return e

class TrainGDX(TrainGDA, TrainGDM):
    """
    Gradient descent with momentum backpropagation and adaptive lr
    
    :Support networks:
        newff (multi-layers perceptron)
    :Рarameters:
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
        lr: float (defaults 0.01)
            learning rate
        adapt: bool (default False)
            type of learning
        lr_inc: float (default 1.05)
            Ratio to increase learning rate
        lr_dec: float (default 0.7)
            Ratio to decrease learning rate
        max_perf_inc:float (default 1.04)
            Maximum performance increase
        mc: float (default 0.9)
            Momentum constant
        rr float (defaults 0.0)
            Regularization ratio
            Must be between {0, 1}
    
    """
    def __init__(self, net, input, target, lr=0.01, adapt=False, lr_inc=1.05, 
                                        lr_dec=0.7, max_perf_inc=1.04, 
                                        mc=0.9, rr =.0):
        """ init gdm"""
        super(TrainGDX, self).__init__(net, input, target, lr, adapt, lr_inc, 
                                        lr_dec, max_perf_inc, rr)
        self.mc = mc
        
    
    
class TrainRprop(TrainGD2):
    """
    Resilient Backpropagation
    
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.07)
            learning rate (init rate)
        adapt bool (default False)
            type of learning
        rate_dec: float (default 0.5)
            Decrement to weight change
        rate_inc: float (default 1.2)
            Increment to weight change
        rate_min: float (default 1e-9)
            Minimum performance gradient
        rate_max: float (default 50)
            Maximum weight change
    
    """
    
    def __init__(self, net, input, target, lr=0.07, adapt=False, 
                    rate_dec=0.5, rate_inc=1.2, rate_min=1e-9, rate_max=50):
        
        super(TrainRprop, self).__init__(net, input, target, lr, adapt)
        self.rate_inc = rate_inc
        self.rate_dec = rate_dec
        self.rate_max = rate_max
        self.rate_min = rate_min
        size = tool.np_size(net)
        self.grad_prev = np.zeros(size)
        self.rate =  np.zeros(size) + lr
    
    def learn(self, net, grad):
    
        prod = grad * self.grad_prev
        # Sign not change
        ind = prod > 0 
        self.rate[ind] *= self.rate_inc
        # Sign change
        ind = prod < 0
        self.rate[ind] *= self.rate_dec
        
        self.rate[self.rate > self.rate_max] = self.rate_max
        self.rate[self.rate < self.rate_min] = self.rate_min
        
        self.x -= self.rate * np.sign(grad)
        self.grad_prev = grad
        return None

class TrainRpropM(TrainRprop):
    """
    Resilient Backpropogation Modified
    (with back-step when grad change sign)
    :Support networks:
        newff (multi-layers perceptron)
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
        lr: float (defaults 0.07)
            learning rate (init rate)
        adapt bool (default False)
            type of learning
        rate_dec: float (default 0.5)
            Decrement to weight change
        rate_inc: float (default 1.2)
            Increment to weight change
        rate_min: float (default 1e-9)
            Minimum performance gradient
        rate_max: float (default 50)
            Maximum weight change
    
    """
    
    def learn(self, net, grad):
    
        prod = grad * self.grad_prev
        # Sign not change
        ind = prod > 0 
        self.rate[ind] *= self.rate_inc
        # Sign change
        ind = prod < 0
        # Back step
        self.x[ind] -= self.rate[ind] * np.sign(grad[ind])
        grad[ind] *= -1
        
        self.rate[ind] *= self.rate_dec
        
        self.rate[self.rate > self.rate_max] = self.rate_max
        self.rate[self.rate < self.rate_min] = self.rate_min
        
        self.x -= self.rate * np.sign(grad)
        self.grad_prev = grad
        return None