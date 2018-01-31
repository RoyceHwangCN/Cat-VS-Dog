#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

DOG_DIR = '/data/python/input/train/dog/'
CAT_DIR = '/data/python/input/train/cat/'
TEST_DIR = '/data/python/input/test/'
train_dogs = [(DOG_DIR + i, 1) for i in os.listdir(DOG_DIR)]
train_cats = [(CAT_DIR + i, 0) for i in os.listdir(CAT_DIR)]
test_images = [(TEST_DIR + i, -1) for i in os.listdir(TEST_DIR)]

train_images = train_dogs + train_cats
random.shuffle(train_images)

ROWS = 64
COLS = 64

def read_image(tuple_set):
    file_path = tuple_set[0]
    label = tuple_set[1]
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_CUBIC), label

CHANNELS = 3

def prep_data(images):
    no_images = len(images)
    data = np.ndarray((no_images, ROWS, COLS, CHANNELS))
    labels = []
    for i, image_file in enumerate(images):
        image, label = read_image(image_file)
        data[i] = image
        labels.append(label)
    return data, labels

x_train, y_train = prep_data(train_images)
x_test, y_test = prep_data(test_images)

print(x_train.shape)
print(y_train.shape)
print(type(x_train))
print(type(y_train))

optimizer = RMSprop(1r=1e-4)
objective = 'binary_crossentropy'

model = Sequential()

#Neural Network Layer1
model.add(Convolution2D(32, (3, 3), padding="same", activation="relu", input_shape=(ROWS, COLS, 3)))
model.add(Convolution2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Neural Network Layer2
model.add(Convolution2D(64, (3, 3), padding="same", activation="relu"))
model.add(Convolution2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Neural Network Layer3
model.add(Convolution2D(128, (3, 3), padding="same", activation="relu"))
model.add(Convolution2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Neural Network Layer4
model.add(Convolution2D(256, (3, 3), padding="same", activation="relu"))
model.add(Convolution2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

nb_epoch = 10
batch_size = 10

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, model='auto')
history = LossHistory()

y_train = np.array(y_train)
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, verbose=0, shuffle=True, callbacks=[history, early_stopping])

model.save(/data/python/cat_dog.h5)
