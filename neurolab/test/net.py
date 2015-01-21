# coding: utf-8

import unittest
import numpy as np
import neurolab as nl
import numpy.random as rand

class TestNet(unittest.TestCase):
    def test_newp(self):
        # Logical &
        input = [[0, 0], [0, 1], [1, 0], [1, 1]]
        target = [[0], [0], [0], [1]]

        # Create net with 2 inputs and 1 neuron
        net = nl.net.newp([[0, 1],[0, 1]], 1)

        # train with delta rule
        # see net.trainf
        error = net.train(input, target, epochs=100, show=0, lr=0.1)
        self.assertEqual(error[-1], 0)
    
    def test_newc(self):
        centr = np.array([[0.2, 0.2], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5]])
        rand_norm = 0.05 * rand.randn(100, 4, 2)
        inp = np.array([centr + r for r in rand_norm])
        inp.shape = (100 * 4, 2)
        rand.shuffle(inp) 

        # Create net with 2 inputs and 4 neurons
        net = nl.net.newc([[0.0, 1.0],[0.0, 1.0]], 4)
        # train with rule: Conscience Winner Take All algoritm (CWTA)
        funcs = [nl.train.train_wta, nl.train.train_cwta]
        for func in funcs:
            net.init()
            error = net.train(inp, epochs=50, show=0)
                
            self.assertLess(error[-1], error[0])
            self.assertLess(np.sum(centr) - np.sum(net.layers[0].np['w']), 0.1)
    
    def test_newlvq(self):
        # Create train samples
        input = np.array([[-3, 0], [-2, 1], [-2, -1], [0, 2], [0, 1], 
                                [0, -1], [0, -2], [2, 1], [2, -1], [3, 0]])
        target = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], 
                                [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])

        # Create network with 2 layers:4 neurons in input layer(Competitive)
        # and 2 neurons in output layer(liner)
        net = nl.net.newlvq(nl.tool.minmax(input), 4, [.6, .4])
        # Train network
        error = net.train(input, target, epochs=100, goal=-1, show=0)
        
        self.assertEqual(error[-1], 0)
    
    def test_newhop(self):
        # N E R O
        target =  [[1,0,0,0,1,
                   1,1,0,0,1,
                   1,0,1,0,1,
                   1,0,0,1,1,
                   1,0,0,0,1],
                  [1,1,1,1,1,
                   1,0,0,0,0,
                   1,1,1,1,1,
                   1,0,0,0,0,
                   1,1,1,1,1],
                  [1,1,1,1,0,
                   1,0,0,0,1,
                   1,1,1,1,0,
                   1,0,0,1,0,
                   1,0,0,0,1],
                  [0,1,1,1,0,
                   1,0,0,0,1,
                   1,0,0,0,1,
                   1,0,0,0,1,
                   0,1,1,1,0]]

        chars = ['N', 'E', 'R', 'O']
        target = np.asfarray(target)
        target[target == 0] = -1

        # Create and train network
        net = nl.net.newhop(target)

        output = net.sim(target)
        
        test =np.asfarray([0,0,0,0,0,
                           1,1,0,0,1,
                           1,1,0,0,1,
                           1,0,1,1,1,
                           0,0,0,1,1])
        test[test==0] = -1
        out = net.sim([test])
        self.assertEqual(out[0].tolist(), target[0].tolist())
    
    def test_newelm(self):
        i1 = np.sin(np.arange(0, 20))
        i2 = np.sin(np.arange(0, 20)) * 2

        t1 = np.ones([1, 20])
        t2 = np.ones([1, 20]) * 2

        input = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
        target = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)

        # Create network with 2 layers
        net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        # Set initialized functions and init
        net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.layers[1].initf= nl.init.InitRand([-0.1, 0.1], 'wb')
        net.init()
        # Train network
        error = net.train(input, target, epochs=100, show=0, goal=0.1)
        
        self.assertLess(error[-1], error[0])
        self.assertLess(error[-1], 0.5)
    
    def test_newhem(self):
        target = [[-1, 1, -1, -1, 1, -1, -1, 1, -1],
                  [1, 1, 1, 1, -1, 1, 1, -1, 1],
                  [1, -1, 1, 1, 1, 1, 1, -1, 1],
                  [1, 1, 1, 1, -1, -1, 1, -1, -1],
                  [-1, -1, -1, -1, 1, -1, -1, -1, -1]]
        input = [[-1, -1, 1, 1, 1, 1, 1, -1, 1],
                 [-1, -1, 1, -1, 1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, 1, -1, -1, 1, -1]]
        # Create and train network
        net = nl.net.newhem(target)
        output = net.sim(target)
        # Test on train samples (must be [0, 1, 2, 3, 4])"
        self.assertEqual(np.argmax(output, axis=0).tolist(), [0, 1, 2, 3, 4])
