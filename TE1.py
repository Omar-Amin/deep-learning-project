import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import keras
from keras import layers
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

sns.heatmap(X_train[0, :, :])

X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))

X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = keras.Sequential()
model.add(layers.Dense(784, activation="relu", input_shape=(784,)))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
# loss="mean_squared_error", metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=30, batch_size=128)


plt.show()

print(10)
