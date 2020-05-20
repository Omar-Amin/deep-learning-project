from myPooling import myPooling as myPool
from myDense import myDense as myDense
from myConvolution2D import myConvolution2D as myConv
from keras import backend as K
from keras.layers import Layer, Flatten, Dense, Conv2D, Conv3D
from keras.models import Model
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import EarlyStopping
import tensorflow as tf

Conv2D()
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
conv = myConv(16, kernel_size=3, activation="relu")(input)
pool = myPool()(conv)
conv = myConv(32, kernel_size=3, activation="relu")(pool)
pool = myPool()(conv)
conv = myConv(64, kernel_size=3, activation="relu")(pool)
pool = myPool()(conv)
flat = Flatten()(pool)
output = myDense(10, activation="softmax")(flat)

model = Model(input, output)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

mcallback = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=1, verbose=2, mode="min"
)

print(model.summary())

model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=30,
    validation_split=0.3,
    callbacks=[mcallback],
)
