import os

from PIL import ImageOps
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np


MODEL_PATH = 'digits.h5'
mode = 'train'

if os.path.exists(MODEL_PATH):
    mode = 'predict'
    print('Loading existing model...')
    model = load_model(MODEL_PATH)
else:
    mode = 'train'
    print('Creating new model')
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


train_images, train_labels = [], []


def train(img, number_value, train_count):
    global train_images, train_labels

    img=ImageOps.invert(img)
    img_array = np.array(img) / 255.0

    train_images.append(img_array)
    train_labels.append(number_value)

    if len(train_images) >= train_count:
        model.fit(np.array(train_images), np.array(train_labels), epochs=5, verbose=0)
        model.save(MODEL_PATH)
        train_images, train_labels = [], []
        return True

    return False


def predict(img):
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    result = np.argmax(prediction)

    return result


def mnist():
    global model, mode, train_images, train_labels

    import random
    import matplotlib.pyplot as plt
    import matplotlib
    import tensorflow as tf

    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)


    plt.rcParams['axes.titlepad'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['image.cmap'] = 'gray'

    SEED_VALUE = 42

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)

    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

    x_valid = x_train_all[:10000]
    x_train = x_train_all[10000:]

    y_valid = y_train_all[:10000]
    y_train = y_train_all[10000:]

    plt.figure(figsize=(6, 5))
    plt.axis(True)
    plt.imshow(x_train[11], cmap='gray')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_train = x_train.astype('float32') / 255.0

    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    model = tf.keras.Sequential()
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    training_result = model.fit(
        x_train, y_train,
        epochs=21,
        batch_size=64,
        validation_data=(x_valid, y_valid)
    )

    print(training_result)

    model.save(MODEL_PATH)
    mode = 'predict'
