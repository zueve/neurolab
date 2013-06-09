import unittest
from trans import TestTrans
from error import TestError

suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(TestTrans))
suite.addTest(unittest.makeSuite(TestError))

unittest.TextTestRunner(verbosity=2).run(suite)
