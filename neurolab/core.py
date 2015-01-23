# -*- coding: utf-8 -*-
"""
Define Core Classes

"""
import numpy as np
from . import tool


class NeuroLabError(Exception):
    pass


class TrainStop(NeuroLabError):
    pass


class Net(object):
    """
    Neural Network class

    :Parameters:
        inp_minmax: minmax: list ci x 2
            Range of input value
        co: int
            Number of output
        layers: list of Layer
            Network layers
        connect: list of list
            Connection scheme of layers*
        trainf: callable
            Train function
        errorf: callable
            Error function with derivative
        
    :Connect format:
        Example 1: for two-layers feed forwad network
            >>> connect = [[-1], # - layer 0 receives the input network signal;
            ...            [0],  # - layer 1 receives the output signal
            ...                  # from the layer 0;
            ...            [1]]  # - the network exit receives the output
            ...                  # signal from the layer 1.

        Example 2: for two-layers Elman network with derivatives:
            >>> connect = [[-1, 0], # - layer 0 receives the input network
            ...                     # signal and output signal from layer 0;
            ...            [0],     # - layer 1 receives the output
            ...                     # signal from the layer 0;
            ...            [1]]     # - the network exit receives the output
            ...                     # signals from the layer 1.

        """

    def __init__(self, inp_minmax, co, layers, connect, trainf, errorf):
        self.inp_minmax = np.asfarray(inp_minmax)
        self.out_minmax = np.zeros([co, 2])
        self.ci = self.inp_minmax.shape[0]
        self.co = co
        self.layers = layers
        self.trainf = trainf
        self.errorf = errorf
        self.inp = np.zeros(self.ci)
        self.out = np.zeros(self.co)
        # Check connect format 
        assert self.inp_minmax.ndim == 2
        assert self.inp_minmax.shape[1] == 2
        if len(connect) != len(layers) + 1:
            raise ValueError("Connect error")
        # Check connect links
        tmp = [0] * len(connect)
        for con in connect:
            for s in con:
                if s != -1:
                    tmp[s] += 1
        for l, c in enumerate(tmp):
            if c == 0 and l != len(layers):
                raise ValueError("Connect error: Lost the signal " +
                                    "from the layer " + str(l - 1))
        self.connect = connect

        # Set inp_minmax for all layers
        for nl, nums_signal in enumerate(self.connect):
            if nl == len(self.layers):
                minmax = self.out_minmax
            else:
                minmax = self.layers[nl].inp_minmax
            ni = 0
            for ns in nums_signal:
                t = self.layers[ns].out_minmax if ns != -1 else self.inp_minmax
                if ni + len(t) > len(minmax):
                    raise ValueError("Connect error: on layer " + str(l - 1))
                minmax[ni: ni + len(t)] = t
                ni += len(t)
            if ni != len(minmax):
                raise ValueError("Connect error: Empty inputs on layer " + 
                                                                    str(l - 1))
        self.init()

    def step(self, inp):
        """
        Simulated step

        :Parameters:
            inp: array like
                Input vector
        :Returns:
            out: array
                Output vector

        """
        #TODO: self.inp=np.asfarray(inp)?
        
        self.inp = inp
        for nl, nums in enumerate(self.connect):
            if len(nums) > 1:
                signal = []
                for ns in nums:
                    s = self.layers[ns].out if ns != -1 else inp
                    signal.append(s)
                signal = np.concatenate(signal)
            else:
                ns = nums[0]
                signal = self.layers[ns].out if ns != -1 else inp
            if nl != len(self.layers):
                self.layers[nl].step(signal)
        self.out = signal
        return self.out

    def sim(self, input):
        """
        Simulate a neural network

        :Parameters:
            input: array like
                array input vectors
        :Returns:
            outputs: array like
                array output vectors
        """
        input = np.asfarray(input)
        assert input.ndim == 2
        assert input.shape[1] == self.ci

        output = np.zeros([len(input), self.co])

        for inp_num, inp in enumerate(input):
            output[inp_num, :] = self.step(inp)

        return output

    def init(self):
        """
        Iinitialization layers

        """
        for layer in self.layers:
            layer.init()

    def train(self, *args, **kwargs):
        """
        Train network
        see net.trainf.__doc__

        """
        return self.trainf(self, *args, **kwargs)

    def reset(self):
        """
        Clear of deley

        """
        self.inp.fill(0)
        self.out.fill(0)
        for layer in self.layers:
            layer.inp.fill(0)
            layer.out.fill(0)

    def save(self, fname):
        """
        Save network on file

        :Parameters:
            fname: file name

        """
        tool.save(self, fname)

    def copy(self):
        """
        Copy network

        """
        import copy
        cnet = copy.deepcopy(self)

        return cnet


class Layer(object):
    """
    Abstract Neural Layer class

    :Parameters:
        ci: int
            Number of inputs
        cn: int
            Number of neurons
        co: int
            Number of outputs
        property: dict
            property: array shape
            example: {'w': (10, 1), 'b': 10}

    """
    def __init__(self, ci, cn, co, property):
        self.ci = ci
        self.cn = cn
        self.co = co
        self.np = {}
        for p, shape in property.items():
            self.np[p] = np.empty(shape)
        self.inp = np.zeros(ci)
        self.out = np.zeros(co)
        # Property must be change when init Layer
        self.out_minmax = np.empty([self.co, 2])
        # Property will be change when init Net
        self.inp_minmax = np.empty([self.ci, 2])
        self.initf = None

    def step(self, inp):
        """ Layer simulation step """
        assert len(inp) == self.ci
        out = self._step(inp)
        self.inp = inp
        self.out = out

    def init(self):
        """ Init Layer random values """
        if type(self.initf) is list:
            for initf in self.initf:
                initf(self)
        elif self.initf is not None:
            self.initf(self)

    def _step(self, inp):
        raise NotImplementedError("Call abstract metod Layer._step")
        
        
class Trainer(object):
    """
    Control of network training
    
    """
    
    def __init__(self, Train, epochs=500, goal=0.01, show=100, **kwargs):
        """
        :Parameters:
            Train: Train instance
                Train algorithm
            epochs: int (default 500)
                Number of train epochs
            goal: float (default 0.01)
                The goal of train
            show: int (default 100)
                Print period
            **kwargs: dict
                other Train parametrs
        
        """
        
        # Sets defaults train params
        self._train_class = Train
        self.defaults = {}
        self.defaults['goal'] = goal
        self.defaults['show'] = show
        self.defaults['epochs'] = epochs
        self.defaults['train'] = kwargs
        if Train.__init__.__defaults__:
            #cnt = Train.__init__.func_code.co_argcount
            cnt = Train.__init__.__code__.co_argcount
            #names = Train.__init__.func_code.co_varnames
            names = Train.__init__.__code__.co_varnames
            vals = Train.__init__.__defaults__
            st = cnt - len(vals)
            for k, v in zip(names[st: cnt], vals):
                if k not in self.defaults['train']:
                    self.defaults['train'][k] = v
        
        self.params = self.defaults.copy()
        self.error = []
    
    def __str__(self):
        return 'Trainer(' + self._train_class.__name__ + ')'
            
    def __call__(self, net, input, target=None, **kwargs):
        """
        Run train process
        
        :Parameters:
            net: Net instance
                network
            input: array like (l x net.ci)
                train input patterns
            target: array like (l x net.co)
                train target patterns - only for train with teacher
            **kwargs: dict
                other Train parametrs
        
        """
        
        self.params = self.defaults.copy()
        self.params['train'] = self.defaults['train'].copy()
        for key in kwargs:
            if key in self.params:
                self.params[key] = kwargs[key]
            else:
                self.params['train'][key] = kwargs[key]
        
        args = []
        input = np.asfarray(input)
        assert input.ndim == 2
        assert input.shape[1] == net.ci
        args.append(input)
        if target is not None:
            target = np.asfarray(target)
            assert target.ndim == 2
            assert target.shape[1] == net.co
            assert target.shape[0] == input.shape[0]
            args.append(target)
        
        def epochf(err, net, *args):
            """Need call on each epoch"""
            if err is None:
                err = train.error(net, *args)
            self.error.append(err)
            epoch = len(self.error)
            show = self.params['show']
            if show and (epoch % show) == 0:
                print("Epoch: {0}; Error: {1};".format(epoch, err))
            if err < self.params['goal']:
                raise TrainStop('The goal of learning is reached')
            if epoch >= self.params['epochs']:
                raise TrainStop('The maximum number of train epochs is reached')
        
        train = self._train_class(net, *args, **self.params['train'])
        Train.__init__(train, epochf, self.params['epochs'])
        self.error = []
        try:
            train(net, *args)
        except TrainStop as msg:
            if self.params['show']:
                print(msg)
        else:
            if self.params['show'] and len(self.error) >= self.params['epochs']:
                print("The maximum number of train epochs is reached")
        return self.error


class Train(object):
    """Base train abstract class"""
    
    def __init__(self, epochf, epochs):
        self.epochf = epochf
        self.epochs = epochs
    
    def __call__(self, net, *args):
        for epoch in range(self.epochs):
            err = self.error(net, *args)
            self.epochf(err, net, *args)
            self.learn(net, *args)
    
    def error(self, net, input, target, output=None):
        """Only for train with teacher"""
        if output is None:
            output = net.sim(input)
        return net.errorf(target, output)
