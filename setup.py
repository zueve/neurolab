""" Setup file for neurolab package """

from distutils.core import setup
import os

def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ''

nl = __import__('neurolab')
version = nl.__version__
status = nl.__status__


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
