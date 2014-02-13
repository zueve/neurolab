#coding: utf-8
""" Setup file for neurolab package """

from distutils.core import setup
import os

def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ''

import neurolab as nl


version = '0.3.1'
status = '4 - Beta'


setup(name='neurolab',
        version=version,
        description='Simple and powerfull neural network library for python',
        long_description = nl.__doc__,
        author='Zuev Evgenij',
        author_email='zueves@gmail.com',
        url='http://neurolab.googlecode.com',
        packages=['neurolab', 'neurolab.train'],
        scripts=[],

        classifiers=(
            'Development Status :: ' + status,
            'Environment :: Console',
            'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
			'Programming Language :: Python :: 2',
			'Programming Language :: Python :: 3',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ),
        license="LGPL-3",
        platforms = ["Any"],
        keywords="neural network, neural networks, neural nets, backpropagation, python, matlab, numpy, machine learning"
    )
