import neurolab as nl
import doctest
import inspect

def load_tests(loader, tests, ignore):
    for m in dir(nl):
        m = eval('nl.' + m)
        if inspect.ismodule(m):
            doctest.testmod(m, verbose=True)
            tests.addTests(doctest.DocTestSuite(m))
    return tests

