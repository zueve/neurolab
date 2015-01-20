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

    def __call__(self, target, output):
        e = target - output
        N = e.size
        v =  np.sum(np.square(e)) / N
        return v

    def deriv(self, target, output):
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
        
        e = target - output
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

    def __call__(self, target, output):
        e = target - output
        v = 0.5 * np.sum(np.square(e))
        return v

    def deriv(self, target, output):
        """
        Derivative of SSE error function

        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        
        return target - output


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

    def __call__(self, target, output):
        e = target - output
        v = np.sum(np.abs(e))
        return v

    def deriv(self, target, output):
        """
        Derivative of SAE error function

        :Parameters:
            e: ndarray
                current errors: target - output
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
        e: ndarray
            current errors: target - output
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
            e: ndarray
                current errors: target - output
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
    From kwecht https://github.com/kwecht/NeuroLab/blob/master/code/error.py
    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases
    :Returns:
        v: float
            Error value
    :Example:
        >>> f = CEE()
        >>> tar = np.array([1,1,0,0])
        >>> out = np.array([0.99,0.01,0.2,0.01])
        >>> x = f(tar,out)
        1.212
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
        n = target.size
        v = - np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
        
        return v

    def deriv(self, target, output):
        """
        y = max(min(y,1),0);
        t = max(min(t,1),0);
        dy = (-t./(y+eps)) + ((1-t)./(1-y+eps));
        
        """
        y = output.copy()
        t = target.copy()
        eps = 0.0
        y[y > (1 - eps)] = 1 - eps
        y[y < eps] = eps
        t[t > (1 - eps)] = 1 - eps
        t[t < eps] = eps
        n = y.size
        #dC/dy = - d/y + (1-d)/(1-y)
        eps = np.spacing(1)
        dy = -t / (y + eps) + (1 - t) / (1 - y + eps)
        
        return dy
        

class CEE2:

    def __call__(self, target, output):
        # Objective term in cost function
        N = target.size
        v = -1.*np.sum(target*np.nan_to_num(np.log(output)) + 
                       (1-target)*np.nan_to_num(np.log(1-output))) / N

        
        return v

    def deriv(self, target, output):
        """
        Derivative of CEE error function
        :Parameters:
            target: ndarray
                target values
            output: ndarray
                network predictions
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        
        """

        N = target.size
        e = -1.*(target - output) / N
        return -e
