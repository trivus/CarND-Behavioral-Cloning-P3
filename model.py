import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from random import random
from os import path

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader, None)
    for line in reader:
        lines.append(line)


cam_correction = .05
correction = [0, cam_correction, -cam_correction]

def generate(ls):
    images = []
    measurements = []
    for line in ls:
        measurement = float(line[3])
        if measurement <= .85 and random() < .7:
            continue
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + filename
            image = cv2.imread(current_path)
            steer = measurement + correction[i]

            images.append(image)
            measurements.append(steer)
            images.append(cv2.flip(image, 1))
            measurements.append(-steer)
    return images, measurements


def generate_train_batch(ls, batch_size=32):
    while 1:
        offset = 0
        for i_batch in range(batch_size):
            x, y = generate(ls[offset: offset+batch_size])
            offset += batch_size
            if offset >= len(ls):
                offset = 0
                print('restart')
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

file_path = './model.h5'
if path.isfile(file_path):
    model = load_model(file_path)
else:
    model = Sequential()
    model.add(Cropping2D(cropping=((75, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x/255.) - .5))
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

tb_callback = TensorBoard(log_dir='log')

train_generator = generate_train_batch(lines, batch_size=100)

#model.fit(X_train, y_train, validation_split=.2, shuffle=True, epochs=2, callbacks=[tb_callback])
model.fit_generator(train_generator, steps_per_epoch=50, epochs=5, callbacks=[tb_callback])


model.save(file_path)

