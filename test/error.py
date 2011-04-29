import unittest
import numpy as np
from neurolab.error import MSE, SSE, SAE

class TestError(unittest.TestCase):
    
    def test_deriv(self):
        vars = np.arange(-100, 100, 2.5)
        test_fcns = [SSE()]
        def diff(f, x, h=1E-6):
            x1 = np.array([x - h])
            x2 = np.array([x])
            r = (f(x2) - f(x1))/h
            return -r
        
        for test_fcn in test_fcns:
            for var in vars:
                m = diff(test_fcn, var)
                v = test_fcn(np.array([var]))
                t = test_fcn.deriv(np.array([v]))[0]
                self.assertAlmostEqual(m, t, 5)
