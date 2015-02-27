# coding: utf-8
import unittest
import numpy as np
from neurolab.trans import TanSig, PureLin, LogSig, HardLim, HardLims, Competitive, SatLin, SatLins, SoftMax, SatLinPrm


class TestTrans(unittest.TestCase):

    def test_tansig(self):
        test_fcn = TanSig()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        from math import tanh
        model = tanh
        m_res = map(model, vars)
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_purelin(self):
        test_fcn = PureLin()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        model = lambda x: x
        m_res = map(model, vars)
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_logsig(self):
        test_fcn = LogSig()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        from math import exp
        model = lambda x: 1 / (1 + exp(-x))
        m_res = map(model, vars)
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_hardlim(self):
        test_fcn = HardLim()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [0, 0, 0, 1, 1]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_satlin(self):
        test_fcn = SatLin()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [0, 0, 0, 0.1, 1.0]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_satlinprm(self):
        # like SatLin
        test_fcn = SatLinPrm(1, 0, 1)
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [0, 0, 0, 0.1, 1.0]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)
        # like SatLins
        test_fcn = SatLinPrm(1, -1, 1)
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [-1.0, -0.5, 0.0, 0.1, 1.0]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_satlins(self):
        test_fcn = SatLins()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [-1.0, -0.5, 0.0, 0.1, 1.0]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_competitive(self):
        test_fcn = Competitive()
        vars = [-2.5, -0.5, 0.0, 0.1, 3.0]
        m_res = [1, 0, 0, 0, 0]
        t_res = test_fcn(np.array(vars)).tolist()
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_softmax(self):
        test_fcn = SoftMax()
        vars = [0, 1, 0.5, -0.5]
        m_res = [1., 4., 2., 1.]
        t_res = np.floor(test_fcn(np.array(vars)) * 10)
        for m, t in zip(m_res, t_res):
            self.assertEqual(m, t)

    def test_deriv(self):
        vars = np.arange(-100, 100, 2.5)
        test_fcns = [
            TanSig(), PureLin(), LogSig(), HardLim(), HardLims(), SatLin(), 
            SatLins(), SatLinPrm(1, 0, 1), SatLinPrm(1, -1, 1), SoftMax()]

        def diff(f, x, h=1E-6):
            x1 = np.array([x - h])
            x2 = np.array([x])
            r = (f(x2)[0] - f(x1)[0]) / h
            return r

        for test_fcn in test_fcns:
            for var in vars:
                m = diff(test_fcn, var)
                v = test_fcn(np.array([var]))
                t = test_fcn.deriv(np.array([var]), v)[0]
                self.assertAlmostEqual(m, t, 5)

    def test_props(self):
        test_fcns = [TanSig(), PureLin(),  # LogSig(),
                     HardLim(), HardLims(), Competitive(), SatLin(), SatLins()]
        vars = np.arange(-1000, 1000, 10)
        for test_fcn in test_fcns:
            self.assertEqual(
                test_fcn.out_minmax[1] >= test_fcn.out_minmax[0],
                True)
            self.assertEqual(
                test_fcn.inp_active[1] >= test_fcn.inp_active[0],
                True)

            for v in vars:
                r = test_fcn(np.array([v]))
                self.assertEqual(
                    test_fcn.out_minmax[0] <= r <= test_fcn.out_minmax[1],
                    True)
