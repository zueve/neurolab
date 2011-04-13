# -*- coding: utf-8 -*-
""" Train error functions with derivatives
    
    :Example:
        >>> msef = MSE()
        >>> x = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> msef(x)
        1.25
        >>> # calc derivative:
        >>> msef.deriv(x[0])
        array([ 1.,  0.])
    
"""

import numpy as np


class MSE():
    """ 
    Mean squared error function 
    
    :Parameters:
        e: ndarray
            current errors: target - output
    :Returns:
        v: float
            Error value
    :Example:
        >>> f = MSE()
        >>> x = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> f(x)
        1.25
        
    """
    
    def __call__(self, e):
        N = e.size
        v =  np.sum(np.square(e)) / N
        return v
        
    def deriv(self, e):
        """
        Derivative of MSE error function
        
        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        :Example:
            >>> f = MSE()
            >>> x = np.array([1.0, 0.0])
            >>> # calc derivative:
            >>> f.deriv(x)
            array([ 1.,  0.])
        
        """
        
        N = len(e)
        d = e * (2 / N)
        return d


class SSE:
    """ 
    Sum squared error function
    
    :Parameters:
        e: ndarray
            current errors: target - output
    :Returns:
        v: float
            Error value
            
    """
    
    def __call__(self, e):
        v = 0.5 * np.sum(np.square(e))
        return v
    
    def deriv(self, e):
        """
        Derivative of SSE error function
        
        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        return e
        

class SAE:
    """ 
    Sum absolute error function
    
    :Parameters:
        e: ndarray
            current errors: target - output
    :Returns:
        v: float
            Error value
    """
    
    def __call__(self, e):
        v = np.sum(np.abs(e))
        return v
    
    def deriv(self, e):
        """
        Derivative of SAE error function
        
        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        d = np.sign(e)
        return d
