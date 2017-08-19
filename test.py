#!/usr/bin/env python

"testing (validating) a trained model"

import numpy as np
from keras.layers import LSTM, Input
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils.np_utils import to_categorical

from PointerLSTM import PointerLSTM

#

n_steps = 3

x_file = 'data/x.csv'
y_file = 'data/y.csv'

split_at = 0
batch_size = 100

hidden_size = 64
weights_file = 'model_weights/model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)

#

x = np.loadtxt(x_file, delimiter=',', dtype=int)
y = np.loadtxt(y_file, delimiter=',', dtype=int)

x = np.expand_dims(x, axis=2)

YY = []
for y_ in y:
    YY.append(to_categorical(y_))
YY = np.asarray(YY)

x_train = x[:split_at]
x_test = x[split_at:]

y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]

seq_len = x.shape[1]

#

print("building model...")
main_input = Input(shape=(seq_len, 1), name='main_input')

encoder = LSTM(output_dim=hidden_size, return_sequences=True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)

print("loading weights from {}...".format(weights_file))
try:
    model.load_weights(weights_file)
except IOError:
    print("no weights file.")
    raise SystemExit

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("testing...")

p = model.predict(x_test[0:1])
p_bin = p.argmax(axis=2)

for y_, p_ in zip(y_test, p)[:5]:
    print("y_test:", y_)
    print("p:     ", p_.argmax(axis=1))
    print()

print("categorical accuracy: ", categorical_accuracy(YY_test, p).eval())
