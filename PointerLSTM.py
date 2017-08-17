# from keras import initializations
import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import _time_distributed_dense


class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def get_initial_states(self, x_input):
        return Recurrent.get_initial_state(self, x_input)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        # init = initializations.get('orthogonal')
        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        # print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>

        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
