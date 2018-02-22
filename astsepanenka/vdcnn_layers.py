from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation


class ConvBlockVDCNN(object):

    def __init__(self, input_shape, num_filters):
        self.model = Sequential()
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def __call__(self, inputs):
        return self.model(inputs)

    