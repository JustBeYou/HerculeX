from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add, Add
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model
import tensorflow as tf # tf de la te fut

import constants
import numpy as np

cnn_filter_num = 128
cnn_first_filter_size = 2
cnn_filter_size = 2
l2_reg = 0.0001
res_layer_num = 20
n_labels = constants.OUTPUT_DIM
value_fc_size = 64
learning_rate = 0.1 # schedule dependent on thousands of steps, every 200 thousand steps, decrease by factor of 10
momentum = 0.9

class Alpha():
    def __init__(self, id=None, **kwargs):
        self.model = self.build_model() # e chiar sub tine fa, tuto
        self.id = id
        self.input_dim = constants.INPUT_DIM

    def build_model(self):
        """
        Builds the full Keras model and returns it.
        """
        in_x = x = Input(shape=constants.INPUT_DIM)

        # (batch, channels, height, width)
        x = Conv2D(filters=cnn_filter_num,   kernel_size=cnn_first_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="input_conv-"+str(cnn_first_filter_size)+"-"+str(cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="policy_conv-1-2")(res_out)

        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)

        # no output for 'pass'
        policy_out = Dense(n_labels, kernel_regularizer=l2(l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="value_conv-1-4")(res_out)

        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(value_fc_size, kernel_regularizer=l2(l2_reg), activation="relu", name="value_dense")(x)

        value_out = Dense(1, kernel_regularizer=l2(l2_reg), activation="tanh", name="value_out")(x)

        model = Model(in_x, [value_out, policy_out], name="hex_model")

        sgd = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        losses = ['categorical_crossentropy', 'mean_squared_error']

        model.compile(loss=losses, optimizer='adam', metrics=['accuracy', 'mae'])

        #model.summary()
        return model

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv1-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv2-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

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
                ret[idx + 1][id] = [2 if el == 1 or el == 2 else 0 for el in row]

        ret[len(ret) - 1] = (np.ones((self.input_dim[0], self.input_dim[1])) * player)

        ret = np.moveaxis(ret, 0, -1)

        return np.reshape(ret, (
        1, *ret.shape))  # np.reshape(ret, (1, self.input_dim[0], self.input_dim[1], self.input_dim[2]))

    def save(self, path):
        #self.model_refresh_without_nan()
        self.model.save(f"{path}/residual.{self.id}.h5")

    def save_smecheros(self, path):
        #self.model_refresh_without_nan()
        self.model.save(path)

    def load(self, path):
        id = path.split(".")[-2]
        self.model = load_model(path)
        self.id = id

    def fit(self, data, labels, epochs, verbose, validation_split, batch_size):
        return self.model.fit(data, labels, epochs=epochs, verbose=verbose,
                              batch_size=batch_size)

    @tf.function
    def predict(self, x):
        return self.model(x, training=True)