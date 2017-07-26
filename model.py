import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from random import random
from sklearn.utils import shuffle
from os import path
from util import *
import argparse


_shear = False
def generate(ls, folder_path, shear):
    '''
    Custom image generator. 
    1. Add left, right images with small adjustment to steering angle
    2. Randomly flip image to balance right/left turns
    :param ls: lines (center, left, right, steer, ...)
    :param folder_path: path containing image files
    :param shear: flag for turning random shear on/off
    :return: modified images and measurements
    '''
    images = []
    measurements = []
    cam_correction = .22
    correction = [0, cam_correction, -cam_correction]

    for line in ls:
        measurement = float(line[3])
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = folder_path + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            steer = measurement + correction[i]
            if shear is True:
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
            x, y = generate(ls[offset: offset+batch_size], img_path, _shear)
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
        # skip first line
        headers = next(reader, None)
        for line in reader:
            lines.append(line)
    lines = shuffle(lines, random_state=1)

    n_train = int(len(lines) * .8)
    train_lines = lines[:n_train]
    valid_lines = lines[n_train:]

    model.compile(loss='mse', optimizer=Adam(lr=5e-4))
    tb_callback = TensorBoard(log_dir='log')
    train_generator = generate_train_batch(train_lines, data_path+'IMG/', batch_size=100)
    valid_generator = generate_train_batch(valid_lines, data_path+'IMG/', batch_size=100)
    model.fit_generator(train_generator, steps_per_epoch=len(lines)/100, epochs=2,
                        validation_data=valid_generator, validation_steps=10,
                        callbacks=[tb_callback])


def main():
    parser = argparse.ArgumentParser(description='Behavior clone model')
    parser.add_argument('--data', help='data dir', dest='data', type=str, default='./data/')
    parser.add_argument('--model', help='model path', dest='model', type=str, default='./model.h5')
    parser.add_argument('--shear', help='Random shear. Use it only for track 1', dest='_shear', action='store_true')
    args = parser.parse_args()
    _model = build_model(args.model, True)
    train_model(_model, args.data)
    _model.save(args.model)

if __name__ == '__main__':
    main()
