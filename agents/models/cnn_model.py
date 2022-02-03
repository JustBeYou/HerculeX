from keras import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
import tensorflow as tf
tf.config.run_functions_eagerly(True)


class CNNModel:
    def __init__(self, board_size):
        input_shape = (board_size, board_size, 1)
        self.model = Sequential()

        self.model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(board_size ** 2, activation='softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def predict(self, input):
        input = input.reshape(1, input.shape[0], input.shape[1], 1)
        return self.model.predict(input)