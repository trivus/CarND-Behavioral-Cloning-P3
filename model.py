import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    # get center image
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: (x/255.) - .5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=5)
model.save('./model.h5')