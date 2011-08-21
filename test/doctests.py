import neurolab as nl
import doctest
import inspect

for m in dir(nl):
    m = eval('nl.' + m)
    if inspect.ismodule(m):
        doctest.testmod(m, verbose=True)
