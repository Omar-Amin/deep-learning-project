from myPooling import myPooling as myPool
from myDense import myDense as myDense
from myConvolution2D import myConvolution2D as myConv
from keras import backend as K
from keras.layers import Layer, Flatten
from keras.models import Model, load_model
import keras
import numpy as np
from keras.utils import np_utils
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import EarlyStopping, ModelCheckpoint

# img processing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import os
import tensorflow as tf


def search(directory, data, labels, label):
    i = 0
    for filename in os.listdir(directory):
        if i % 25 == 0:
            print(i)
        i = i + 1
        if filename.endswith(".jpeg"):
            img = load_img(
                directory + filename, grayscale=True, target_size=(224, 224),
            )
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(label)
        elif filename.endswith(".txt"):
            # filename.readLline()
            data.append(np.genfromtxt(directory + filename))
            labels.append(label)


def make_model():
    data = []
    y_train = []

    search("project_2020/filtered/NORMAL/", data, y_train, 0)
    search("project_2020/filtered/PNEUMONIA/", data, y_train, 1)
    data = np.asarray(data)
    y_train = np.asarray(y_train)
    data = data / 255.0
    y_train = np_utils.to_categorical(y_train)

    # Now we start defining our model. Note that the input shape is (28,28,1)
    input = Input(shape=(224, 224, 1))
    conv = myConv(24, kernel_size=4, activation="relu")(input)
    pool = myPool()(conv)
    conv = myConv(12, kernel_size=4, activation="relu")(pool)
    pool = myPool()(conv)
    conv = myConv(6, kernel_size=4, activation="relu")(pool)
    pool = myPool()(conv)
    flat = Flatten()(pool)
    output = myDense(units=2, activation="softmax")(flat)

    model = Model(input, output)
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    mcallback = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=1, verbose=2, mode="min"
    )

    print(model.summary())

    model.fit(
        data,
        y_train,
        batch_size=32,
        epochs=30,
        validation_split=0.3,
        callbacks=[mcallback, ModelCheckpoint("model2.h5")],
        shuffle=True,
    )

    # Save model
    model.save("model1.h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def evaluate_model():
    data = []
    y_test = []

    search("project_2020/encoded/NORMAL/", data, y_test, 0)
    search("project_2020/encoded/PNEUMONIA/", data, y_test, 1)

    data = np.asarray(data)
    y_test = np.asarray(y_test)
    data = data / 255.0
    y_test = np_utils.to_categorical(y_test)

    model = load_model(
        filepath="model2.h5",
        custom_objects={
            "myConvolution2D": myConv,
            "myPooling": myPool,
            "myDense": myDense,
        },
    )
    model.evaluate_model(data, y_test)


make_model()
evaluate_model()
