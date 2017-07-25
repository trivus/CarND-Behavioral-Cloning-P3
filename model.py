import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from random import random
from sklearn.utils import shuffle
from os import path
from util import *


def generate(ls, folder_path):
    '''
    Custom image generator. 
    1. Undersample close-to 0 steering data randomly to 20% for balanced data set.
    2. Add left, right images with small adjustment to steering angle
    3. Add flipped image to balance right/left turns
    5. Apply random brightness
    4. Resize images
    :param ls: lines (center, left, right, steer, ...)
    :param folder_path: path containing image files
    :return: modified images and measurements
    '''
    images = []
    measurements = []
    cam_correction = .22
    correction = [0, cam_correction, -cam_correction]

    for line in ls:
        measurement = float(line[3])
        #if abs(measurement) <= .08 and random() <= .2:
        #    continue
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = folder_path + filename
            image = cv2.imread(current_path)
            # turned random brightness off
            # image = random_brightness(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # image = image[60:-20, ...]
            # image = cv2.resize(image, (160, 40), cv2.INTER_AREA)
            steer = measurement + correction[i]
            image, steering = random_shear(image, measurement, shear_range=40)

            if random() > .5:
                images.append(image)
                measurements.append(steer)
            else:
                images.append(cv2.flip(image, 1))
                measurements.append(-steer)
    return images, measurements


def generate_train_batch(ls, img_path, batch_size=32):
    while 1:
        for offset in range(0, len(ls), batch_size):
            x, y = generate(ls[offset: offset+batch_size], img_path)
            offset += batch_size
            yield np.array(x), np.array(y)


def build_model(file_path, old_model=False):
    '''
    modified nvidia model
    :param file_path: path to save/load model
    :param old_model: if true, load previous model and train from there
    :return:
    '''
    if path.isfile(file_path) and old_model is True:
        model = load_model(file_path)
    else:
        model = Sequential()
        # decreased top crop for 2nd track
        model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: (x / 255.) - .5))
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
    return model


def train_model(model, data_path):
    lines = []
    with open(data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        for line in reader:
            lines.append(line)
    lines = shuffle(lines, random_state=1)
    model.compile(loss='mse', optimizer=Adam(lr=5e-4))
    tb_callback = TensorBoard(log_dir='log')
    train_generator = generate_train_batch(lines, data_path+'IMG/', batch_size=100)
    #model.fit(X_train, y_train, validation_split=.2, shuffle=True, epochs=2, callbacks=[tb_callback])
    model.fit_generator(train_generator, steps_per_epoch=len(lines)/100, epochs=2, callbacks=[tb_callback])


MODEL_PATH = './model.h5'
DATA_PATH = './data/'
_model = build_model(MODEL_PATH, True)
train_model(_model, DATA_PATH)
_model.save(MODEL_PATH)

# https://github.com/naokishibuya/car-behavioral-cloning/blob/master/model.py