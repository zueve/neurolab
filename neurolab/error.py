# -*- coding: utf-8 -*-
""" Train error functions with derivatives

    :Example:
        >>> msef = MSE()
        >>> x = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> msef(x, 0)
        1.25
        >>> # calc derivative:
        >>> msef.deriv(x[0], 0)
        array([ 1.,  0.])

"""

import numpy as np


class MSE():
    """
    Mean squared error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value
    :Example:
        >>> f = MSE()
        >>> x = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> f(x, 0)
        1.25

    """

    def __call__(self, target, output):
        e = target - output
        N = e.size
        v =  np.sum(np.square(e)) / N
        return v

    def deriv(self, target, output):
        """
        Derivative of MSE error function

        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        :Example:
            >>> f = MSE()
            >>> x = np.array([1.0, 0.0])
            >>> # calc derivative:
            >>> f.deriv(x, 0)
            array([ 1.,  0.])

        """
        
        e = target - output
        N = len(e)
        d = e * (2 / N)
        return d


class SSE:
    """
    Sum squared error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value

    """

    def __call__(self, target, output):
        e = target - output
        v = 0.5 * np.sum(np.square(e))
        return v

    def deriv(self, target, output):
        """
        Derivative of SSE error function

        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        
        return target - output


class SAE:
    """
    Sum absolute error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value
    
    """

    def __call__(self, target, output):
        e = target - output
        v = np.sum(np.abs(e))
        return v

    def deriv(self, target, output):
        """
        Derivative of SAE error function

        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        e = target - output
        d = np.sign(e)
        return d


class MAE:
    """
    Mean absolute error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value
    
    """

    def __call__(self, target, output):
        e = target - output
        v = np.sum(np.abs(e)) / e.size
        return v

    def deriv(self, target, output):
        """
        Derivative of SAE error function

        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        e = target - output
        d = np.sign(e) / e.size
        return d
        

class CEE:
    """
    Cross-entropy error function.
    For use when targets in {0,1}
    
    C = -sum( t * log(o) + (1 - t) * log(1 - o))
    
    Thanks kwecht https://github.com/kwecht
    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value
    
    """

    def __call__(self, target, output):
        # Objective term in cost function
        y = output.copy()
        t = target.copy()
        eps = np.spacing(1)
        y[y > (1 - eps)] = 1 - eps
        y[y < eps] = eps
        t[t > (1 - eps)] = 1 - eps
        t[t < eps] = eps
        v = - np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
        v /= t.size
        return v

    def deriv(self, target, output):
        """
        Derivative of CEE error function
        
        dC/dy = - t/o + (1 - t) / (1 - o)

        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        
        """
        y = output.copy()
        t = target.copy()
        eps = 0.0
        y[y > (1 - eps)] = 1 - eps
        y[y < eps] = eps
        t[t > (1 - eps)] = 1 - eps
        t[t < eps] = eps
        #dC/dy = - d/y + (1-d)/(1-y)
        eps = np.spacing(1)
        dy = t / (y + eps) - (1 - t) / (1 - y + eps)
        dy /= t.size
        return dy
