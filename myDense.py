from keras import backend as K
from keras.layers import *
from keras.utils import conv_utils
from keras import initializers, activations, regularizers
import tensorflow as tf
from keras.utils import serialize_keras_object


class myDense(Layer):
    def __init__(
        self,
        units=32,
        bias=True,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        activation=None,
        **kwargs
    ):
        self.units = units
        self.bias = bias
        self.weight_initializer = initializers.get(weight_initializer)
        self.weight_regularizer = regularizers.get(weight_regularizer)
        self.activation = activations.get(activation)
        super(myDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            trainable=True,
        )
        if self.bias:
            self.b = self.add_weight(
                name="b", shape=(self.units,), initializer="zeros", trainable=True
            )
        else:
            self.b = 0

        super(myDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super(myDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "bias": self.bias,
                "weight_initializer": initializers.serialize(self.weight_initializer),
                "weight_regularizer": regularizers.serialize(self.weight_regularizer),
            }
        )

        return config
