# coding: utf-8
import unittest
import numpy as np
from neurolab.error import MSE, SSE, SAE, MAE, CEE

class TestError(unittest.TestCase):

    def test_deriv(self):
        vars = np.arange(-100, 100, 2.5)
        vars = np.random.randn(100,2) * 50
        test_fcns = [MSE(), SSE(), SAE(), MAE()]
        def diff2(f, x, y, h=1E-6):
            x2 = np.array([x + h])
            x1 = np.array([x])
            y1 = np.array([y])
            y2 = np.array([y])
            r = (f(x2, y2) - f(x1, y1)) / h
            return r    

        for test_fcn in test_fcns:
            for var in vars:
                d1 = diff2(test_fcn, var[0], var[1])
                d2 = test_fcn.deriv(np.array([var[0]]), np.array([var[1]]))[0]
                #print var, v1, v2
                if not isinstance(test_fcns, SAE) \
                    and var[0] != 0.0 and var[1] != 0:
                    self.assertAlmostEqual(d1, d2, 5)
