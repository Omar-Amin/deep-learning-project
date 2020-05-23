from keras import backend as K
from keras.layers import *
from keras.models import Model
import numpy as np


class myPooling(Layer):
    """Abstract class for different pooling 2D layers.
    """

    # "max" for max, "avg" for average
    def __init__(
        self, pool_size=(2, 2), strides=None, padding="valid", pool_mode="max", **kwargs
    ):
        super(myPooling, self).__init__(**kwargs)
        self.pool_mode = pool_mode
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)

    def call(self, inputs):
        output = K.pool2d(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            "channels_last",
            pool_mode=self.pool_mode,
        )
        return output

    def compute_output_shape(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]
        rows = conv_utils.conv_output_length(
            rows, self.pool_size[0], self.padding, self.strides[0]
        )
        cols = conv_utils.conv_output_length(
            cols, self.pool_size[1], self.padding, self.strides[1]
        )
        return (input_shape[0], rows, cols, input_shape[3])

    def get_config(self):
        config = super(myPooling, self).get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "padding": self.padding,
                "strides": self.strides,
                "pool_mode": self.pool_mode,
            }
        )
        return config
