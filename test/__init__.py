import unittest
from test.trans import TestTrans
from test.error import TestError

suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(TestTrans))
suite.addTest(unittest.makeSuite(TestError))

unittest.TextTestRunner(verbosity=2).run(suite)