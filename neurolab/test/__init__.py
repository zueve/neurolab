import unittest
#import neurolab.test.doctests as doctests
from . import doctests
from .trans import TestTrans
from .error import TestError
from .tool import TestTool

suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromModule(doctests))
suite.addTest(unittest.makeSuite(TestTrans))
suite.addTest(unittest.makeSuite(TestError))
suite.addTest(unittest.makeSuite(TestTool))

def test():
    unittest.TextTestRunner(verbosity=2).run(suite)
    import neurolab as nl
    print('Neurolab version {}'.format(nl.__version__))
