import unittest
from trans import TestTrans

suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(TestTrans))

unittest.TextTestRunner(verbosity=2).run(suite)