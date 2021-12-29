import tensorflow as tf
tf.config.run_functions_eagerly(True)

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers

import numpy as np

class ResidualModel:
    def __init__(self, regularizer, learning_rate, input_dim, output_dim, hidden_layers, momentum):
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.momentum = momentum

        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(hidden_layers)
        self.model = self.build_model()

    def softmax_cross_entropy_with_logits(y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
        where = tf.equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0)
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

        return loss

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

        #for layer in self.hidden_layers[1:]:
            #model = self.residual_layer(model, layer['filters'], layer['kernel_size'])

        value_head = self.value_head(model)
        policy_head = self.policy_head(model)

        model_final = Model(inputs=input, outputs=[value_head, policy_head])
        model_final.compile(loss={'value_head': 'mean_squared_error',
                                  'policy_head': self.softmax_cross_entropy_with_logits},
                            optimizer=SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                            loss_weights={'value_head': 0.5, 'policy_head': 0.5})

        return model_final

    def fit(self, data, labels, epochs, verbose, validation_split, batch_size):
        return self.model.fit(data, labels, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)

    def transform_input(self, state):
        return np.reshape(state[0], (1, self.input_dim[0], self.input_dim[1], 1))

    def predict(self, x):
        return self.model.predict(x)

    def save(self, version):
        self.model.save('saved_models/' + version + '.h5')

    def load(self, version):
        return load_model('save_models' + version + '.h5')
