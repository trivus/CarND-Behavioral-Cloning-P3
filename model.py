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

lines = []
data_path = './data2/'

with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader, None)
    for line in reader:
        lines.append(line)
lines = shuffle(lines, random_state=1)


def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_bright = 1.0 + 0.1 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def random_shear(image, steering, shear_range):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    #    print('dx',dx)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 10.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering

    return image, steering


def generate(ls):
    '''
    Custom image generator. 
    1. Undersample close-to 0 steering data randomly to 20% for balanced data set.
    2. Add left, right images with small adjustment to steering angle
    3. Add flipped image to balance right/left turns
    5. Apply random brightness
    4. Resize images
    :param ls: lines (center, left, right, steer, ...)
    :return: modified images and measurements
    '''
    images = []
    measurements = []
    cam_correction = .3
    correction = [0, cam_correction, -cam_correction]

    for line in ls:
        measurement = float(line[3])
        #if abs(measurement) <= .08 and random() <= .2:
        #    continue
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = data_path + 'IMG/' + filename
            image = cv2.imread(current_path)
            #image = image[60:-20, ...]
            #image = cv2.resize(image, (160, 40), cv2.INTER_AREA)
            image = random_brightness(image)
            steer = measurement + correction[i]

            #image, steering = random_shear(image, measurement, shear_range=40)

            if random() > .5:
                images.append(image)
                measurements.append(steer)
            else:
                images.append(cv2.flip(image, 1))
                measurements.append(-steer)
    return images, measurements


def generate_train_batch(ls, batch_size=32):
    while 1:
        offset = 0
        x, y = generate(ls[offset: offset+batch_size])
        offset += batch_size
        if offset >= len(ls):
            offset = 0
            print('\nrestart')
        yield np.array(x), np.array(y)


# for line in lines:
#     measurement = float(line[3])
#     if measurement <= .85 and random() < .5:
#         continue
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         current_path = './data/IMG/' + filename
#         image = cv2.imread(current_path)
#         steer = measurement + correction[i]
#
#         images.append(image)
#         measurements.append(steer)
#         images.append(cv2.flip(image, 1))
#         measurements.append(-steer)

        # if random() > .5:
        #     images.append(image)
        #     measurements.append(steer)
        # else:
        #     images.append(cv2.flip(image, 1))
        #     measurements.append(-steer)

# X_train = np.array(images)
# y_train = np.array(measurements)

file_path = './model2.h5'
load = True
if path.isfile(file_path) and load is True:
    model = load_model(file_path)
else:
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x/255.) - .5))
    model.add(Conv2D(24, (5,5), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=5e-4))

tb_callback = TensorBoard(log_dir='log')

train_generator = generate_train_batch(lines, batch_size=100)

#model.fit(X_train, y_train, validation_split=.2, shuffle=True, epochs=2, callbacks=[tb_callback])
model.fit_generator(train_generator, steps_per_epoch=len(lines)/100, epochs=1, callbacks=[tb_callback])


model.save(file_path)

