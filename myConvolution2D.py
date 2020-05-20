from keras import backend as K
from keras.layers import *
from keras.models import Model
import numpy as np


class myConvolution2D(Layer):
    def __init__(
        self,
        filters=1,
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
