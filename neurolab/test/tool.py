# coding: utf-8
import unittest
import numpy as np
import neurolab as nl
import tempfile


class TestTool(unittest.TestCase):

    def test_save_load(self):

        nets = [nl.net.newff([[-7, 7]], [5, 1]),
                nl.net.newc([[0.0, 1.0]], 10),
                nl.net.newelm([[-2, 2]], [10, 1],
                              [nl.trans.TanSig(), nl.trans.PureLin()]),
                nl.net.newhem(np.random.random([10, 1])),
                nl.net.newhop(np.random.random([10, 1])),
                nl.net.newlvq([[-1, 1]], 4, [.6, .4]),
                nl.net.newp([[0, 1]], 1),
                nl.net.newgrnn([[-3, 1], [-5, 2]], [[1], [2]], [[-3], [-5]], 1),
                nl.net.newpnn([[-2, 0], [-2, 0]], [[-2, -2], [0, 0]], [[1, 0], [0, 1]], 2)
                ]
        for net in nets:
            input = np.random.random([10, 1])
            fname = tempfile.gettempdir() + '/' + 'temp.neurolab'
            net.save(fname)
            output = net.sim(input)
            net2 = nl.tool.load(fname)
            output2 = net2.sim(input)
            self.assertEqual(output.tolist(), output2.tolist())
