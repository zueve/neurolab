# -*- coding: utf-8 -*-
""" 
Functions of initialization  layers 

"""


import numpy as np
        
        
def init_rand(layer, min=-0.5, max=0.5, init_prop='w'):
    """
    Initialize the specified property of the layer 
    random numbers within specified limits
    
    :Parameters:
        layer:
            Initialized layer
        min: float (default -0.5)
            minimum value after the initialization
        max: float (default 0.5)
            maximum value after the initialization
        init_prop: str (default 'w')
            name of initialized property, must be in layer.np
    
    """
    
    if init_prop not in layer.np:
        raise ValueError('Layer not have attibute "' + init_prop + '"')
    layer.np[init_prop] = np.random.uniform(min, max, layer.np[init_prop].shape)

def initwb_reg(layer):
    """ 
    Initialize weights and bias 
    in the range defined by the activation function (transf.inp_active)
    
    """
    active = layer.transf.inp_active[:]
    
    if np.isinf(active[0]):
        active[0] = -100.0
    
    if np.isinf(active[1]):
        active[1] = 100.0
    
    min = active[0] / (2 * layer.cn)
    max = active[1] / (2 * layer.cn)

    init_rand(layer, min, max, 'w')
    if 'b' in layer.np:
        init_rand(layer, min, max, 'b')

def initwb_reg2(layer):
    """ 
    Initialize weights and bias 
    in the range defined by the activation function (transf.inp_active)
    
    """
    active = layer.transf.inp_active[:]
    
    if np.isinf(active[0]):
        active[0] = -10.0
    
    if np.isinf(active[1]):
        active[1] = 10.0
    
    ci = layer.ci if 'b' not in layer.np else layer.ci + 1
    dist = float(active[1] - active[0]) / ci
    
    inp = layer.inp_minmax
    out = [0, 0]
    out[0] = active[0] + (active[1] - active[0]) / 2 - dist / 2
    out[1] = out[0] + dist
    
    max_p = (out[1] - out[0]) / (inp[:, 1] - inp[:, 0])
    max_n = - max_p
    
    w = np.random.randint(2, size=layer.np['w'].shape)
    layer.np['w'][w==0] = np.random.uniform(max_n, 0, layer.np['w'].shape)[w==0]
    layer.np['w'][w==1] = np.random.uniform(0, max_p, layer.np['w'].shape)[w==1]

    if 'b' in layer.np:
        layer.np['b'] = np.random.uniform(out[0], out[1], layer.np['b'].shape)

def initwb_reg3(layer):
    """ 
    Initialize weights and bias 
    in the range defined by the activation function (transf.inp_active)
    
    """
    active = layer.transf.inp_active[:]
    
    if np.isinf(active[0]):
        active[0] = -10.0
    
    if np.isinf(active[1]):
        active[1] = 10.0
    
    ci = layer.ci if 'b' not in layer.np else layer.ci + 1
    dist = float(active[1] - active[0]) / ci
    
    inp = layer.inp_minmax
    out = [0, 0]
    out[0] = active[0] + (active[1] - active[0]) / 2 - dist / 2
    out[1] = out[0] + dist
    
    max_p = (out[1] - out[0]) / (inp[:, 1] - inp[:, 0])
    max_n = - max_p
    
    w = np.random.randint(2, size=layer.np['w'].shape)
    layer.np['w'][w==0] = np.random.uniform(max_n, 0, layer.np['w'].shape)[w==0]
    layer.np['w'][w==1] = np.random.uniform(0, max_p, layer.np['w'].shape)[w==1]

    k = 0.7 * layer.cn**(1./layer.ci)
    layer.np['w'] *= k
    
    if 'b' in layer.np:
        layer.np['b'] = np.random.uniform(out[0], out[1], layer.np['b'].shape)
        layer.np['b'] *= k


def initwb_nw(layer):
    """ 
    Initialize weights and bias with Nguyen-Widrow rule 
    In the range defined by the activation function (transf.inp_active)
    
    """
    
    inp = np.asfarray(layer.inp_minmax)
    active = np.asfarray(layer.transf.inp_active)
    
    if np.isinf(active[0]):
        active[0] = -10.0
    
    if np.isinf(active[1]):
        active[1] = 10.0
    
    k = 1. / layer.cn / layer.ci
    w = k * np.ones([layer.cn, layer.ci])
    #b = k * (np.linspace(-1, 1, layer.cn) if layer.cn > 1 else np.zeros(1))# - (inp.max()+inp.min())/2.
    scale = (active[1] - active[0]) / (np.max(inp) - np.min(inp))
    w *= scale
    if layer.cn > 1:
        #b = k*np.linspace(-1, 1, layer.cn) #* (inp.max() + inp.min())*1.8#*layer.cn/2 +1.9 #- k * (inp.max() + inp.min()) / 2
        #b = np.linspace(active[0] + (active[1] - active[0])/layer.cn/2, active[1] - (active[1] - active[0])/layer.cn/2, layer.cn)
        b = k*np.linspace(active[0] - scale * inp.max(), active[1] - scale * inp.min(), layer.cn) - 4.5
    else:
        b = np.zeros(1) - k * (inp.max() + inp.min()) / 2
    print 'k', k
    
    # Scale
    #b = np.array([-1., 1.])
    #print 'b', layer.np['b']
    #print 'b2', b
    #print 's', scale
    #b *= scale 
    #b += active[1] + active[0]
    
    #x = 0.5 * (active[1] - active[0])
    #y = 0.5 * (active[1] + active[0])
    
    #w = w * x
    #b = b * x + y
    
    layer.np['w'][:] = w
    if 'b' in layer.np:
        layer.np['b'] = b


class InitRand:
    """
    Initialize the specified properties of the layer 
    random numbers within specified limits
    
    """
    def __init__(self, minmax, init_prop):
        """
        :Parameters:
            minmax: list of float
                [min, max] init range
            init_prop: list of dicts
                names of initialized propertis. Example ['w', 'b']
        
        """
        self.min = minmax[0]
        self.max = minmax[1]
        self.properties = init_prop
    
    def __call__(self, layer):
        for property in self.properties:
            init_rand(layer, self.min, self.max, property)
        return


def init_zeros(layer):
    """
    Set all layer properties of zero
    
    """
    for k in layer.np:
        layer.np[k].fill(0.0)
    return


def midpoint(layer):
    """
    Sets weight to the center of the input ranges
    
    """
    mid = layer.inp_minmax.mean(axis=1)
    for i, w in enumerate(layer.np['w']):
        layer.np['w'][i] = mid.copy()
    return
    