#!/usr/bin/env python

"""
generate data for training, save to a file
shape: n_examples x n_elements x n_features
the task is to order three elements in each row by the sum of their elements
"""

import numpy as np
import random

x_file = 'data/x_sums.csv'
y_file = 'data/y_sums.csv'

n_examples = 10000
n_steps = 3
n_features = 2

max_int = 20

# sums may repeat
x = np.random.randint( max_int, size = ( n_examples, n_steps, n_features ))
s = x.sum( axis = 2 )
y = np.argsort( s )

np.savetxt( x_file, x.reshape( x.shape[0], -1 ), delimiter = ',', fmt = '%d' )
np.savetxt( y_file, y, delimiter = ',', fmt = '%d' )
