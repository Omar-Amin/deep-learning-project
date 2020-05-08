# Note here that I have prepended tensorflow. to all the imports.
# The reason for this is the new version of Tensorflow causing an incompatibility with the Keras version we are using, when trying to use TensorBoard.
# The new version of Tensorflow has a Keras version built in. For normal use, you can use either one, except if you want to use TensorBoard.
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, concatenate
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model

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
input_1 = Input(shape=(28, 28, 1))
conv2d_1 = Conv2D(16, kernel_size=3, activation="relu")(input_1)
conv2d_2 = Conv2D(16, kernel_size=3, activation="relu")(input_1)
max_pooling2d_1 = MaxPooling2D()(conv2d_1)
max_pooling2d_2 = MaxPooling2D()(conv2d_2)
add_1 = Add()([max_pooling2d_1, max_pooling2d_2])
conv2d_3 = Conv2D(16, kernel_size=3, activation="relu")(add_1)
flatten_1 = Flatten()(conv2d_3)
dense_1 = Dense(16, activation="relu")(flatten_1)
dense_2 = Dense(16, activation="relu")(dense_1)
concatenate_1 = concatenate([dense_2, flatten_1])
# As we have 10 classes in our data, the last dense layer must have 10 units, and a softmax activation function.
dense_3 = Dense(10, activation="softmax")(concatenate_1)

model = Model(inputs=input_1, outputs=dense_3)

# Because we have a multi-class problem, we use categorical crossentropy as our loss function
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

# Here we plot our model to make sure it has the same structure as the task directed.
plot_model(model, to_file="model.png")

# Time to define our callbacks.
# As per the instruction, we use ModelCheckpoint to save the best version of our model.
# We also set up TensorBoard. The argument we pass to TensorBoard says where TensorBoard should save the logs of our training progress.
callbacks = [ModelCheckpoint("best_model.h5"), TensorBoard("./logs")]

# Because we pass TensorBoar as a callback, we can see graphs of our metrics live, by calling "tensorboard --logdir logs/" in a terminal and going to http://localhost:6006/.
# See the attached image.
model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=10,
    validation_split=0.3,
    callbacks=callbacks,
)

# Now that we've built the non-sensical model and trained it, it's time to make a sensical version.
# We will use the typical sandwich of convolution -> pooling, and in the end we flatten everything and use a dense layer.
# This is the bread and butter of image classification networks.
input = Input(shape=(28, 28, 1))
conv = Conv2D(16, kernel_size=3, activation="relu")(input)
pool = MaxPooling2D()(conv)
conv = Conv2D(32, kernel_size=3, activation="relu")(pool)
pool = MaxPooling2D()(conv)
conv = Conv2D(64, kernel_size=3, activation="relu")(pool)
pool = MaxPooling2D()(conv)
flat = Flatten()(pool)
output = Dense(10, activation="softmax")(flat)

model = Model(input, output)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=10,
    validation_split=0.3,
    callbacks=callbacks,
)

# Train on 42000 samples, validate on 18000 samples
# Epoch 1/10
# 42000/42000 [==============================] - 16s 376us/sample - loss: 0.2583 - accuracy: 0.9206 - val_loss: 0.1025 - val_accuracy: 0.9678
# Epoch 2/10
# 42000/42000 [==============================] - 17s 399us/sample - loss: 0.0923 - accuracy: 0.9718 - val_loss: 0.0854 - val_accuracy: 0.9743
# Epoch 3/10
# 42000/42000 [==============================] - 19s 442us/sample - loss: 0.0686 - accuracy: 0.9790 - val_loss: 0.0724 - val_accuracy: 0.9784
# Epoch 4/10
# 42000/42000 [==============================] - 22s 525us/sample - loss: 0.0552 - accuracy: 0.9829 - val_loss: 0.0804 - val_accuracy: 0.9770
# Epoch 5/10
# 42000/42000 [==============================] - 25s 592us/sample - loss: 0.0472 - accuracy: 0.9860 - val_loss: 0.0635 - val_accuracy: 0.9824
# Epoch 6/10
# 42000/42000 [==============================] - 26s 619us/sample - loss: 0.0406 - accuracy: 0.9877 - val_loss: 0.0587 - val_accuracy: 0.9844
# Epoch 7/10
# 42000/42000 [==============================] - 25s 596us/sample - loss: 0.0360 - accuracy: 0.9895 - val_loss: 0.0623 - val_accuracy: 0.9832
# Epoch 8/10
# 42000/42000 [==============================] - 25s 589us/sample - loss: 0.0322 - accuracy: 0.9906 - val_loss: 0.0604 - val_accuracy: 0.9849
# Epoch 9/10
# 42000/42000 [==============================] - 26s 623us/sample - loss: 0.0287 - accuracy: 0.9919 - val_loss: 0.0630 - val_accuracy: 0.9849
# Epoch 10/10
# 42000/42000 [==============================] - 27s 644us/sample - loss: 0.0258 - accuracy: 0.9927 - val_loss: 0.0717 - val_accuracy: 0.9837


# Because we used ModelCheckpoint, the best version of our model has been saved, which was after the 9th epoch, so we load this version.
model = load_model("best_model.h5")

# Finally, we get an estimate of our generalization error by evaluating our network on our testing dataset.
model.evaluate(x_test, y_test)

# 10000/10000 [==============================] - 2s 173us/sample - loss: 0.0699 - accuracy: 0.9855
# Our accuracy on the test dataset is 98.55%
