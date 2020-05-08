from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import *
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import EarlyStopping


class myDense(Layer):
    def __init__(self, units=32, input_dim=32, activation=None, **kwargs):
        self.units = units
        self.input_dim = input_dim
        self.activation = activations.get(activation)
        super(myDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(self.units,), initializer="constant", trainable=True
        )

        super(myDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        print("input: ", inputs)
        print("weigths: ", self.w)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super(myDense, self).get_config()
        config.update({"units": self.units})
        return config


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000, 784)) / 255.0
X_test = X_test.reshape((10000, 784)) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras.Sequential()
model.add(myDense(32, activation="relu", input_shape=(784,)))
model.add(myDense(32, activation="relu"))
model.add(myDense(10, activation="softmax"))

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
mcallback = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="min"
)

print(model.summary())

# go overall the dataset 30 times, og 30% af data bliver brugt som validation
history = model.fit(X_train, y_train, epochs=4, batch_size=16, validation_split=0.3)

print(model.evaluate(X_test, y_test))
# x = tf.ones((2, 2))
# layer = myDense(21, 24, activation="lort")
# y = layer(x)
# print(y)
# [0.17176559518257564, 0.9594]
