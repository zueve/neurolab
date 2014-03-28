# coding: utf-8
import unittest
import numpy as np
from neurolab.error import MSE, SSE, SAE, MAE

class TestError(unittest.TestCase):

    def test_deriv(self):
        vars = np.arange(-100, 100, 2.5)
        test_fcns = [MSE(), SSE(), SAE(), MAE()]
        def diff(f, x, h=1E-6):
            x2 = np.array([x + h])
            x1 = np.array([x])
            r = (f(x2) - f(x1))/h
            return r

        for test_fcn in test_fcns:
            for var in vars:
                d1 = diff(test_fcn, var)
                d2 = test_fcn.deriv(np.array([var]))[0]
                #print var, v1, v2
                if not isinstance(test_fcns, SAE) and var != 0.0:
                    self.assertAlmostEqual(d1, d2, 5)
