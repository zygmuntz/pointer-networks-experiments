#!/usr/bin/env python

"testing (validating) a trained model"

import pickle
import numpy as np

from keras.models import Model
from keras.layers import LSTM, Input
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score as accuracy

from PointerLSTM import PointerLSTM

#

x_file = 'data/x_sums_test.csv'
y_file = 'data/y_sums_test.csv'

split_at = 0
batch_size = 100

hidden_size = 100
weights_file = 'model_weights/model_weights_sums_{}.hdf5'.format( hidden_size )

n_steps = 3
n_features = 2

#

x = np.loadtxt( x_file, delimiter = ',', dtype = int )
y = np.loadtxt( y_file, delimiter = ',', dtype = int )

x = x.reshape( x.shape[0], n_steps, -1 )
assert( x.shape[-1] == n_features )

YY = []
for y_ in y:
	YY.append(to_categorical(y_))
YY = np.asarray(YY)

x_train = x[:split_at]
x_test = x[split_at:]

y_train = y[:split_at]
y_test = y[split_at:]

YY_train = YY[:split_at]
YY_test = YY[split_at:]

#

print("building model...")
main_input = Input( shape=( x.shape[1], x.shape[2] ), name='main_input' )

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model( input=main_input, output=decoder )

print( "loading weights from {}...".format( weights_file ))
try:
	model.load_weights( weights_file )
except IOError:
	print( "no weights file." )

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
			  
print( 'testing...')

p = model.predict( x_test )

for y_, p_ in zip( y_test, p )[:5]:
	print "y_test:", y_
	print "p:     ", p_.argmax( axis = 1 )
	print

print "categorical accuracy: ", categorical_accuracy( YY_test, p ).eval()

