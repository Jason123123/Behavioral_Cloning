import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    del(lines[0])

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def flip_image(images, measurements):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(-measurement)
    return np.array(augmented_images), np.array(augmented_measurements)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            angles = []
            path_header = "./data/"

            for batch_sample in batch_samples:
                # generate new training data using different camera
                steering_center = float(batch_sample[3])
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_center = np.array(Image.open(path_header + batch_sample[0].strip()))
                image_left = np.array(Image.open(path_header + batch_sample[1].strip()))
                image_right = np.array(Image.open(path_header + batch_sample[2].strip()))

                images.extend([image_center, image_left, image_right])
                angles.extend([steering_center, steering_left, steering_right])

            X_train, y_train = flip_image(images, angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,activation="elu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5,activation="elu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,activation="elu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
exit()



