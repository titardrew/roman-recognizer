from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization


class ConvNetA(Sequential):
    weights_path = 'weights/conv1.ckpt'

    def __init__(self):
        super().__init__()
        self.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28)))
        self.add(BatchNormalization(axis=-1))
        self.add(Activation('relu'))
        self.add(Conv2D(32, (3, 3)))
        self.add(BatchNormalization(axis=-1))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(64, (3, 3)))
        self.add(BatchNormalization(axis=-1))
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(BatchNormalization(axis=-1))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Flatten())

        # Fully connected layer
        self.add(Dense(128))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Dropout(0.2))
        self.add(Dense(10))

        self.add(Activation('softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def setup_weights(self):
        self.load_weights(self.weights_path)