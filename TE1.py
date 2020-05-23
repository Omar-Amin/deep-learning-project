from myPooling import myPooling as myPool
from myDense import myDense as myDense
from myConvolution2D import myConvolution2D as myConv
from keras import backend as K
from keras.layers import Layer, Flatten, Dropout
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
from keras.preprocessing.image import ImageDataGenerator
import os
from mySequence import mySequence as mySeq
import random


def search(directory, data, labels, label):
    i = 0
    for filename in os.listdir(directory):
        if i % 200 == 0:
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
            matrix = np.expand_dims(np.genfromtxt(directory + filename), axis=-1)
            data.append(matrix)
            labels.append(label)


def make_model():
    data = []
    y_train = []

    search("project_2020/filtered/NORMAL/", data, y_train, 0)
    search("project_2020/filtered/PNEUMONIA/", data, y_train, 1)

    # shuffle at the beginning
    tmp = list(zip(data, y_train))
    random.shuffle(tmp)
    data, y_train = zip(*tmp)

    data = np.asarray(data)
    y_train = np.asarray(y_train)
    data = data / 255.0
    y_train = np_utils.to_categorical(y_train)

    # Now we start defining our model. Note that the input shape is (28,28,1)
    input = Input(shape=(224, 224, 1))
    conv = myConv(32, kernel_size=3, strides=2, activation="relu")(input)
    conv = myConv(32, kernel_size=3, activation="relu")(conv)
    pool = myPool()(conv)
    conv = myConv(64, kernel_size=3, strides=2, activation="relu")(input)
    conv = myConv(64, kernel_size=3, activation="relu")(conv)
    pool = myPool()(conv)
    conv = myConv(128, kernel_size=3, strides=2, activation="relu")(input)
    conv = myConv(128, kernel_size=3, activation="relu")(conv)
    pool = myPool()(conv)
    drop = Dropout(0.2)(pool)

    flat = Flatten()(drop)
    dense = myDense(units=256, bias=True, activation="relu")(flat)
    drop = Dropout(0.4)(dense)
    dense = myDense(units=86, bias=True, activation="relu")(drop)
    drop = Dropout(0.3)(dense)
    dense = myDense(units=32, bias=True, activation="relu")(drop)
    drop = Dropout(0.15)(dense)

    output = myDense(units=2, bias=True, activation="softmax")(drop)

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
        epochs=30,
        validation_split=0.3,
        callbacks=[mcallback, ModelCheckpoint("model3.h5")],
    )

    return model


def evaluate_model(model):
    test = mySeq("project_2020/encoded", 15)

    print(model.evaluate_generator(test))


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model3.h5")
    print("Saved model to disk")
    return model


def load_model_from_disk(path):
    print("Loading model...")
    return load_model(
        path,
        custom_objects={
            "myConvolution2D": myConv,
            "myPooling": myPool,
            "myDense": myDense,
        },
    )


model = make_model()
save_model(model)
model.save("model3.h5")

new_model = load_model_from_disk("model3.h5")
evaluate_model(new_model)

# model = make_model()
# # save_model(model)
# model.save("model1")

# new_model = load_model_from_disk("model1")

# test_input = np.random.random((128, 32))
# np.testing.assert_allclose(
#     model.predict(test_input), new_model.predict(test_input)
# )

# new_model.fit(test_input, test_target)
# evaluate_model(new_model)

# Retrieve the config
# config = newModel.get_config()

# At loading time, register the custom objects with a `custom_object_scope`:
# custom_objects = {
#     "myConvolution2D": myConv,
#     "myPooling": myPool,
#     "myDense": myDense,
# }
# with keras.utils.custom_object_scope(custom_objects):
#     new_model = keras.Model.from_config(config)
#     evaluate_model(new_model)
