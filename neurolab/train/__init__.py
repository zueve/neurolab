# -*- coding: utf-8 -*-
"""
Train algorithms based  gradients algorihms
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

Train algorithms for LVQ networks
=================================
.. autofunction:: train_lvq

Delta rule
==========

.. autofunction:: train_delta

"""

import gd, spo, wta, lvq, delta

def trainer(Train):
    """ Trainner init """
    from neurolab.core import Trainer
    
    c = Trainer(Train)
    c.__doc__ = Train.__doc__
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

train_wta = trainer(wta.TrainWTA)
train_cwta = trainer(wta.TrainCWTA)
train_lvq = trainer(lvq.TrainLVQ)
train_delta = trainer(delta.TrainDelta)