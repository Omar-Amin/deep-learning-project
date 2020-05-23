from keras import backend as K
from keras.layers import *
from keras.utils import conv_utils
from keras import initializers, activations, regularizers
from keras.models import Model
import numpy as np


class myConvolution2D(Layer):
    def __init__(
        self,
        filters=1,
        kernel_size=(3, 3),
        activation=None,
        kernel_initializer="glorot_uniform",
        padding="valid",
        strides=1,
        dilation_rate=(1, 1),
        bias=True,
        **kwargs,
    ):
        self.filters = filters
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.activation = activations.get(activation)
        self.dilation_rate = dilation_rate
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, "kernel_size")
        self.bias = bias
        super(myConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_size + (input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
        )
        if self.bias:
            self.b = self.add_weight(
                name="b", shape=(self.filters,), initializer="zeros", trainable=True
            )

        super(myConvolution2D, self).build(input_shape)

    def call(self, x):
        output = K.conv2d(
            x,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )
        if self.bias:
            output = K.bias_add(output, self.b, "channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
            )
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = super(myConvolution2D, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias": self.bias,
            }
        )
        return config
