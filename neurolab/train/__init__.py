# -*- coding: utf-8 -*-
"""
Train algorithms based  gradients algorithms
===========================================

.. autofunction:: train_gd
.. autofunction:: train_gdm
.. autofunction:: train_gda
.. autofunction:: train_gdx
.. autofunction:: train_rprop

Train algorithms based on Winner Take All - rule
================================================
.. autofunction:: train_wta
.. autofunction:: train_cwta

Train algorithms based on spipy.optimize
========================================
.. autofunction:: train_bfgs
.. autofunction:: train_cg
.. autofunction:: train_ncg

Train algorithms for LVQ networks
=================================
.. autofunction:: train_lvq

Delta rule
==========

.. autofunction:: train_delta

"""

from . import gd, spo, wta, lvq, delta
import functools

def trainer(Train, **kwargs):
    """ Trainner init """
    from neurolab.core import Trainer
    #w = functools.wraps(Train)
    #c = w(Trainer(Train))
    c = Trainer(Train, **kwargs)
    c.__doc__ = Train.__doc__
    c.__name__ = Train.__name__
    c.__module__ = Train.__module__
    return c

# Initializing mains train functors
train_gd = trainer(gd.TrainGD)
#train_gd2 = trainer(gd.TrainGD2)
train_gdm = trainer(gd.TrainGDM)
train_gda = trainer(gd.TrainGDA)
train_gdx = trainer(gd.TrainGDX)
train_rprop = trainer(gd.TrainRprop)
#train_rpropm = trainer(gd.TrainRpropM)

train_bfgs = trainer(spo.TrainBFGS)
train_cg = trainer(spo.TrainCG)
train_ncg = trainer(spo.TrainNCG)

train_wta = trainer(wta.TrainWTA)
train_cwta = trainer(wta.TrainCWTA)
train_lvq = trainer(lvq.TrainLVQ)
train_delta = trainer(delta.TrainDelta)
