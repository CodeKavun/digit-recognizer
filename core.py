import os

from PIL import ImageOps
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

import numpy as np


MODEL_PATH = 'digits.h5'

if os.path.exists(MODEL_PATH):
    print('Loading existing model...')
    model = load_model(MODEL_PATH)
else:
    print('Creating new model')
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


train_images, train_labels = [], []


def train(img, number_value, train_count) -> bool:
    global train_images, train_labels

    img = ImageOps.invert(img)
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
    img_array = np.array(img) / 255.0

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    result = np.argmax(prediction)

    return result
