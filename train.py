import os
import tensorflow as tf
import os
import glob
import numpy as np
import time
from cv2 import cv2

import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

DATA_DIR = 'train_30_3'
NAME = DATA_DIR + '-cnn-256x1-512x1-dense_64-' + str(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

def load_data(directory):
    '''loads the images into an array suitable for tensorflow models'''

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    data_path = os.path.join(os.getcwd(), directory)
    training_data = []
    for letter in alphabet:
        letter_path = os.path.join(data_path, letter)
        class_num = alphabet.index(letter)
        for filename in glob.glob(letter_path + "/*.jpg"):
            img_array = cv2.imread(filename, 0)
            new_array = cv2.resize(img_array, (64, 64))
            training_data.append([new_array, class_num])
    return training_data


directory = os.path.join('data', DATA_DIR)
data = load_data(directory)


random.shuffle(data)

train_x = []
train_y = []

for features, label in data:
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x).reshape(-1, 64, 64, 1)
train_y = np.array(train_y)

train_x = train_x/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(26))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(train_x, train_y, batch_size=10, epochs=10, validation_split=0.1, callbacks=[tensorboard])

model_name = NAME + '.model'
model.save(model_name)


