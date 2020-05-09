from keras import backend as K
from keras.layers import *
from keras.models import Model
import numpy as np
import tensorflow as tf

# fjern
from keras import backend as K
from keras.layers import Layer
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import *
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import EarlyStopping


class myConvolution2D(Layer):
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        activation=None,
        padding="valid",
        strides=1,
        **kwargs,
    ):
        self.filters = filters
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.activation = activations.get(activation)
        self.rank = 2
        if type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        super(myConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_size + (input_shape[-1], self.filters),
            initializer="glorot_uniform",
        )

        super(myConvolution2D, self).build(input_shape)

    def call(self, x):
        output = K.conv2d(x, self.kernel, strides=self.strides, padding=self.padding)
        if self.activation is not None:
            output = self.activation(output)
        self.op_shape = output.shape
        return output

    def compute_output_shape(self, input_shape):
        new_output_shape = []
        for i in range(len(self.op_shape)):
            new_output_shape.append(self.op_shape[i].value)
        return tuple(new_output_shape)


# First we import the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Now we have to reshape the data. The reason for this is that a Conv2D layer expects a 4D tensor.
# The first is the batch dimension, which Keras will handle for us.
# Then we have the x,y coordinate for every pixel in the image.
# The last dimension is the color dimension. The MNIST dataset is monochrome, so it has only 1 color dimension
# Lastly, we divide by 255. to normalize the values to be between [0,1]
x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
x_test = x_test.reshape(10000, 28, 28, 1) / 255.0

# As with the FFN, we convert the labels to a one-hot encoded vector.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Now we start defining our model. Note that the input shape is (28,28,1)
input = Input(shape=(28, 28, 1))
conv = myConvolution2D(16, kernel_size=3, strides=3, activation="relu")(input)
# pool = MaxPooling2D()(conv)
conv = myConvolution2D(32, kernel_size=3, activation="relu")(conv)
# pool = MaxPooling2D()(conv)
conv = myConvolution2D(64, kernel_size=3, activation="relu")(conv)
# pool = MaxPooling2D()(conv)
flat = Flatten()(conv)
output = Dense(10, activation="softmax")(flat)

model = Model(input, output)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.3)
