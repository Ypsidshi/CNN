python get_training_images.py 50 up

import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
GESTURE_CATEGORIES=6
base_model = Sequential()
base_model.add(SqueezeNet(input_shape=(225, 225, 3), include_top=False))
base_model.add(Dropout(0.5))
base_model.add(Convolution2D(GESTURE_CATEGORIES, (1, 1), padding='valid'))
base_model.add(Activation('relu'))
base_model.add(GlobalAveragePooling2D())
base_model.add(Activation('softmax'))
base_model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.train()

model.save("gesture-model05_20.h5")


from keras.models import load_model
import cv2
import numpy as np
img = cv2.imread('скрин.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (225, 225))
model = load_model("gesture-base-rmsprop-model05_20.h5")
prediction = model.predict(np.array([img]))