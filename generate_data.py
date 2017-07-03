#!/usr/bin/env python

"""
generate simple data for training, save to a file
format: x,y: 4,5,1, 1,2,0
"""

import numpy as np
import random

n_examples = 10000
n_steps = 8

arange = 10

x_file = 'data/x_{}.csv'.format( n_steps )
y_file = 'data/y_{}.csv'.format( n_steps )

#

# no repeating numbers within a sequence
x = np.arange( arange ).reshape( 1, -1 ).repeat( n_examples, axis = 0 )
x = np.apply_along_axis( np.random.permutation, 1, x )
x = x[:,:n_steps]

y = np.argsort( x, axis = 1 )

np.savetxt( x_file, x, delimiter = ',', fmt = '%d' )
np.savetxt( y_file, y, delimiter = ',', fmt = '%d' )
