import numpy as np
from scipy.misc import imresize,imshow
from imageio import imread
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def init():
    # [32C3-BN]*2-[32C5S2-BN]-0.4d-[64C3-BN]*2-[64C5S2-BN]-0.4d-F-256D-10D 20E
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(32, 5, padding='same',strides=2, activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(64, 5, padding='same',strides=2, activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.load_weights("./models/model_weights.h5")
    print("Loaded Model from disk")

    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model