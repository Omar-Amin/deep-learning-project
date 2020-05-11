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
        if strides is None:
            self.strides = pool_size
        else:
            self.strides = strides
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.padding = conv_utils.normalize_padding(padding)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        output = K.pool2d(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            "channels_last",
            pool_mode=self.pool_mode,
        )
        self.op_shape = output.shape
        return output

    def compute_output_shape(self, input_shape):
        new_output_shape = []
        for i in range(len(self.op_shape)):
            new_output_shape.append(self.op_shape[i].value)
        return tuple(new_output_shape)
