# -*- coding: utf-8 -*-
""" 
Transfer function with derivatives

:Example:
    >>> import numpy as np
    >>> f = TanSig()
    >>> x = np.linspace(-5,5,100)
    >>> y = f(x)
    >>> df_on_dy = f.deriv(x, y) # calc derivative
    >>> f.out_minmax    # list output range [min, max]
    [-1, 1]
    >>> f.inp_active    # list input active range [min, max]
    [-2, 2]
"""

import numpy as np


class TanSig:
    """
    Hyperbolic tangent sigmoid transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            The corresponding hyperbolic tangent values.
    :Example:
        >>> f = TanSig()
        >>> f([-np.Inf, 0.0, np.Inf])
        array([-1.,  0.,  1.])
    """
    # output range
    out_minmax = [-1, 1]
    # input active range
    inp_active = [-2, 2]
    
    def __call__(self, x):
        return np.tanh(x)
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function TanSig

        """
        return 1.0 - np.square(y)


class PureLin:
    """
    Linear transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            copy of x
    :Example:
        >>> import numpy as np
        >>> f = PureLin()
        >>> x = np.array([-100., 50., 10., 40.])
        >>> f(x).tolist()
        [-100.0, 50.0, 10.0, 40.0]
        
    """
    
    out_minmax = [-np.Inf, np.Inf]
    inp_active = [-np.Inf, np.Inf]
    
    def __call__(self, x):
        return x.copy()
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function PureLin
        
        """
        return np.ones_like(x)


class LogSig:
    """
    Logarithmic sigmoid transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            The corresponding  logarithmic sigmoid values.
    :Example:
        >>> f = LogSig()
        >>> x = np.array([-np.Inf, 0.0, np.Inf])
        >>> f(x).tolist()
        [0.0, 0.5, 1.0]
        
    
    """
    
    out_minmax = [0, 1]
    inp_active =[-4, 4]
    
    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function LogSig
        
        """
        
        return y * (1 - y)


class HardLim:
    """
    Hard limit transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            may take the following values: 0, 1
    
    :Example:
        >>> f = HardLim()
        >>> x = np.array([-5, -0.1, 0, 0.1, 100])
        >>> f(x)
        array([ 0.,  0.,  0.,  1.,  1.])
        
    """
    
    out_minmax = [0, 1]
    inp_active = [0, 0]
    
    def __call__(self, x):
        return (x > 0) * 1.0
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function HardLim
        
        """
        return np.zeros_like(x)


class HardLims:
    """
    Symmetric hard limit transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            may take the following values: -1, 1
    :Example:
        >>> f = HardLims()
        >>> x = np.array([-5, -0.1, 0, 0.1, 100])
        >>> f(x)
        array([-1., -1., -1.,  1.,  1.])
        
    """
    
    out_minmax = [-1, 1]
    inp_active = [0, 0]
    
    def __call__(self, x):
        return (x > 0) * 2.0 - 1.0
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function HardLims
        
        """
        return np.zeros_like(x)


class Competitive:
    """
    Competitive transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            may take the following values: 0, 1
            1 if is a minimal element of x, else 0
    :Example:
        >>> f = Competitive()
        >>> f([-5, -0.1, 0, 0.1, 100])
        array([ 1.,  0.,  0.,  0.,  0.])
        >>> f([-5, -0.1, 0, -6, 100])
        array([ 0.,  0.,  0.,  1.,  0.])
        
    """
    
    out_minmax = [0, 1]
    inp_active = [-np.Inf, np.Inf]
    
    def __call__(self, dist):
        r = np.zeros_like(dist)
        min = np.argmin(dist)
        r[min] = 1.0
        return r

class SatLins:
    """
    Symmetric saturating linear transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            -1 if x < -1; x if -1 <= x <= 1; 1 if x >1
    :Example:
        >>> f = SatLins()
        >>> x = np.array([-5, -1, 0, 0.1, 100])
        >>> f(x)
        array([-1. , -1. ,  0. ,  0.1,  1. ])
        
    """
    
    out_minmax = [-1, 1]
    inp_active = [-1, 1]
    
    def __call__(self, x):
        y = x.copy()
        y[y < -1] = -1
        y[y > 1] = 1
        return y
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function SatLins
        
        """
        d = np.zeros_like(x)
        d[(x > -1) & (x < 1) ] = 1
        
        return d

class SatLin:
    """
    Saturating linear transfer function
    
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            0 if x < 0; x if 0 <= x <= 1; 1 if x >1
    :Example:
        >>> f = SatLin()
        >>> x = np.array([-5, -0.1, 0, 0.1, 100])
        >>> f(x)
        array([ 0. ,  0. ,  0. ,  0.1,  1. ])
        
    """
    
    out_minmax = [0, 1]
    inp_active = [0, 1]
    
    def __call__(self, x):
        y = x.copy()
        y[y < 0] = 0
        y[y > 1] = 1
        return y
    
    def deriv(self, x, y):
        """ 
        Derivative of transfer function SatLin
        
        """
        
        d = np.zeros_like(x)
        d[(x > 0) & (x < 1) ] = 1
        
        return d