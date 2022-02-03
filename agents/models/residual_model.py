import tensorflow as tf

import constants
import numpy as np

import datetime

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss

class ResidualModel:
    def __init__(self, regularizer, learning_rate, input_dim, output_dim, hidden_layers, momentum, id = 'noid'):
        self.id = id

        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.momentum = momentum

        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(hidden_layers)
        self.model = self.build_model()
        #self.model.call = tf.function(self.model.call, experimental_relax_shapes=True)

    def residual_layer(self, input_block, filters, kernel_size):
        model = self.conv_layer(input_block, filters, kernel_size)

        model = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding='same',
                       use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.regularizer))(model)
        model = BatchNormalization(axis=1)(model)
        model = add([input_block, model])
        model = LeakyReLU()(model)

        return model

    def conv_layer(self, model, filters, kernel_size):
        model = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding='same',
                       use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.regularizer))(model)
        model = BatchNormalization(axis=1)(model)
        model = LeakyReLU()(model)

        return model

    def value_head(self, model):
        model = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_last", padding='same', use_bias=False
                       , activation='linear', kernel_regularizer=regularizers.l2(self.regularizer))(model)
        model = BatchNormalization(axis=1)(model)
        model = LeakyReLU()(model)
        model = Flatten()(model)

        model = Dense(20, use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.regularizer))(
            model)

        model = LeakyReLU()(model)

        model = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(self.regularizer),
                      name='value_head')(model)

        return model

    def policy_head(self, model):
        model = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_last", padding='same', use_bias=False,
                       activation='linear', kernel_regularizer=regularizers.l2(self.regularizer))(model)
        model = BatchNormalization(axis=1)(model)
        model = LeakyReLU()(model)
        model = Flatten()(model)
        model = Dense(self.output_dim, use_bias=False, activation='linear',
                      kernel_regularizer=regularizers.l2(self.regularizer), name='policy_head')(model)

        return model

    def build_model(self):
        input = Input(shape=self.input_dim)

        model = self.conv_layer(input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
        model = self.residual_layer(model, self.hidden_layers[1]['filters'], self.hidden_layers[1]['kernel_size'])

        for layer in self.hidden_layers[1:]:
            model = self.residual_layer(model, layer['filters'], layer['kernel_size'])

        value_head = self.value_head(model)
        policy_head = self.policy_head(model)

        model_final = Model(inputs=input, outputs=[value_head, policy_head])
        model_final.compile(loss={'value_head': 'mean_squared_error',
                                  'policy_head': softmax_cross_entropy_with_logits},
                            optimizer=Adam(learning_rate=self.learning_rate),
        #SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                            loss_weights={'value_head': 0.5, 'policy_head': 0.5})
        return model_final

    def compile(self):
        self.model.compile(loss={'value_head': 'mean_squared_error',
                                  'policy_head': softmax_cross_entropy_with_logits},
                            optimizer=Adam(learning_rate=self.learning_rate),
                           #SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                            loss_weights={'value_head': 0.5, 'policy_head': 0.5})

    def fit(self, data, labels, epochs, verbose, validation_split, batch_size):
        return self.model.fit(data, labels, epochs=epochs, verbose=verbose,
                              batch_size=batch_size)

    def transform_input(self, game_state, player):
        # idk why this code is so ugly but I was unable to find something prettier or faster
        # TODO: improve this or Sad :(
        '''
        ret0 = []
        ret1 = []
        for state in game_state:
            temp0 = []
            temp1 = []
            for row in state:
                temp0.append([2 if el == 0 or el == 2 else 1 for el in row])
                temp1.append([2 if el == 1 or el == 2 else 0 for el in row])
            ret0.append(temp0)
            ret1.append(temp1)
        ret = np.append(ret0, ret1)
        ret = np.append(ret, (np.ones((constants.BOARD_SIZE, constants.BOARD_SIZE)) * player))

        return np.reshape(ret, (7, self.input_dim[0], self.input_dim[1], 1))
        '''

        ret = np.zeros(shape=(3, self.input_dim[0], self.input_dim[1]))
        game_state = game_state[-1:]

        for idx, state in enumerate(game_state):
            for id, row in enumerate(state):
                ret[idx][id] = [2 if el == 0 or el == 2 else 1 for el in row]
                ret[idx+1][id] = [2 if el == 1 or el == 2 else 0 for el in row]

        ret[len(ret) - 1] = (np.ones((self.input_dim[0], self.input_dim[1])) * player)

        ret = np.moveaxis(ret, 0, -1)

        return np.reshape(ret, (1, *ret.shape))  # np.reshape(ret, (1, self.input_dim[0], self.input_dim[1], self.input_dim[2]))

    @tf.function
    def predict(self, x):
        return self.model(x, training=True)

    def model_refresh_without_nan(self):
        import numpy as np
        valid_weights = []
        for l in self.model.get_weights():
            if np.isnan(l).any():
                valid_weights.append(np.nan_to_num(l))
                print("!!!!!", l)
            else:
                valid_weights.append(l)
        self.model.set_weights(valid_weights)

    def save(self, path):
        #self.model_refresh_without_nan()
        self.model.save(f"{path}/residual.{self.id}.h5")

    def save_smecheros(self, path):
        #self.model_refresh_without_nan()
        self.model.save(path)

    def load(self, path):
        id = path.split(".")[-2]
        self.model = load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        self.compile()
        self.id = id

        #self.model_refresh_without_nan()