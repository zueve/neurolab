# -*- coding: utf-8 -*-
"""
The module contains the basic layers architectures

"""
from core import Layer
import init
import trans

import numpy as np


class Perceptron(Layer):
    """
    Perceptron Layer class
    
    :Parameters:
        ci: int
            Number of input
        cn: int
            Number of neurons
        transf: callable
            Transfer function
    
    :Example:
        >>> import neurolab as nl
        >>> # create layer with 2 inputs and 4 outputs(neurons)
        >>> l = Perceptron(2, 4, nl.trans.PureLin())
    
    """
    
    def __init__(self, ci, cn, transf):

        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'b': cn})
        
        self.transf = transf
        if not hasattr(transf, 'out_minmax'):
            test = np.asfarry([-1e100, -100, -10, -1, 0, 1, 10, 100, 1e100])
            val = self.transf(test)
            self.out_minmax = np.array([val.min(), val.max()] * self.co)
        else:
            self.out_minmax = np.asfarray([transf.out_minmax] * self.co)
        # default init function
        self.initf = init.initwb_reg
        self.s = np.zeros(self.cn)

    def _step(self, inp):
        self.s = np.sum(self.np['w'] * inp, axis=1)
        self.s += self.np['b']
        return self.transf(self.s)
     

class Competitive(Layer):
    """ 
    Competitive Layer class
    
    :Parameters:
        ci: int
            Number of input
        cn: int
            Number of neurons
        distf: callable default(euclidean)
            Distance function
    
    """
    
    def __init__(self, ci, cn, distf=None):
        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'conscience': cn})
        self.transf = trans.Competitive()
        self.initf = init.midpoint
        self.out_minmax[:] = np.array([self.transf.out_minmax] * cn)
        self.np['conscience'].fill(1.0)
        def euclidean(A, B):
            """
            Euclidean distance function.
            See scipi.spatial.distance.cdist()
            
            :Example:
                >>> import numpy as np
                >>> euclidean(np.array([0,0]), np.array([[0,1], [0, 5.5]])).tolist()
                [1.0, 5.5]
                
            """
            return np.sqrt(np.sum(np.square(A-B) ,axis=1))
        
        self.distf = euclidean
    
    def _step(self, inp):

        d = self.distf(self.np['w'], inp.reshape([1,len(inp)]))
        self.last_dist = d
        out = self.transf(self.np['conscience'] * d)
        return out