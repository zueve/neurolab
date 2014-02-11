import unittest
import doctests
from trans import TestTrans
from error import TestError

#suite = unittest.TestSuite()
suite = unittest.TestLoader().loadTestsFromModule(doctests)
suite.addTest(unittest.makeSuite(TestTrans))
suite.addTest(unittest.makeSuite(TestError))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
