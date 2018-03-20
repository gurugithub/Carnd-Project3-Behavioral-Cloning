# Code by Guru Shetto for P3 CARND-TERM1
# March 6th, 2017 Added dropouts after each convnet based on initial review and recommended by reviewer to reduce overfitting
# -------Imports--------
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt


# --------code-------
# This allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # correction for left and right cameras. Didnt tweak much and stuck to 0.2 with no issues
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        # shuffle to remove any bias
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
        
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                #print (current_path)
                    if (batch_sample[3] != 'steering'):
                        image = cv2.imread(current_path)
                        images.append(image)
                        if (i == 1):
                            measurement = float(batch_sample[3]) + correction
                        elif (i == 2):
                            measurement = float(batch_sample[3]) - correction
                        else:
                            measurement = float(batch_sample[3])

                        measurements.append(measurement)

            augmented_images, augmentated_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmentated_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmentated_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmentated_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

            
            
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Basically the below model is from nvidia https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

# didnt want to reinvent the wheel. Works really well!

# Initialize
model = Sequential()
# Normalization
model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160,320,3)))
# Crop image to remove distractions
model.add(Cropping2D(cropping=((70,25), (0,0))))
# strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))

model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
# non-strided convolution with a 3×3 kernel size in the final two convolutional layers.
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.2))

model.add(Flatten())
# four fully connected layers, leading to a final output control value angle of steering
model.add(Dense(100))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# adam optimizer is used instead of manual tuning
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 7, verbose=1)
# saving the model
model.save('./model/model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

