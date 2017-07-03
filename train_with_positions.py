#!/usr/bin/env python

"""
order integer sequences of length given by n_steps
adding integer position indicator to each number
maybe it will help the network
"""

import pickle
import numpy as np

from keras.models import Model
from keras.layers import LSTM, Input
from keras.utils.np_utils import to_categorical

from PointerLSTM import PointerLSTM

#

n_steps = 5

x_file = 'data/x_{}.csv'.format( n_steps )
y_file = 'data/y_{}.csv'.format( n_steps )

split_at = 9000
batch_size = 100

hidden_size = 64
weights_file = 'model_weights/model_weights_with_positions_{}_steps_{}.hdf5'.format( n_steps, hidden_size )


#

x = np.loadtxt( x_file, delimiter = ',', dtype = int )
y = np.loadtxt( y_file, delimiter = ',', dtype = int )

x = np.expand_dims( x, axis = 2 )

# prep position indicators

n = np.arange( n_steps )
nn = np.tile( n, ( x.shape[0], 1 ))
nn = np.expand_dims( nn, axis = 2 )

xx = np.dstack(( x, nn ))

YY = []
for y_ in y:
	YY.append(to_categorical(y_))
YY = np.asarray(YY)

x_train = xx[:split_at]
x_test = xx[split_at:]

y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]

assert( n_steps == x_train.shape[1] )
n_features = x_train.shape[2]

print( "sequence length is {}, {} features at each step.".format( n_steps, n_features ))

#

print( "building model..." )
main_input = Input( shape=( n_steps, n_features ), name='main_input' )

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model( input=main_input, output=decoder )

print( "loading weights from {}...".format( weights_file ))
try:
	model.load_weights( weights_file )
except IOError:
	print( "no weights file, starting anew." )

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

			  
print( 'training and saving model weights each epoch...' )

validation_data = ( x_test, YY_test )

while True:

	history = model.fit( x_train, YY_train, nb_epoch = 1, batch_size = batch_size, 
		validation_data = validation_data )

	p = model.predict( x_test )

	for y_, p_ in zip( y_test, p )[:5]:
		print "y_test:", y_
		print "p:     ", p_.argmax( axis = 1 )
		print

	model.save_weights( weights_file )
