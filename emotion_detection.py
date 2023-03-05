# %%
import os
import random

import cv2  ## pip install opencv-python
import numpy as np  ## pim install numpy
import tensorflow
import tensorflow as tf  # pip install tensorflow-gpu
from keras import layers
from tensorflow import keras

DataDirectory = "archive/test/"  # This is for the training dataset

classes = ["0", "1", "2", "3", "4", "5", "6"]  ## list of classes => exact name of your folder

training_data = []  ## data array


def create_training_data():
    """
    This is used to create training data by resizing it for
    imagenet to undertand the code
    """
    #  Reading all images
    for category in classes:
        # looping all classes on list
        # getting the path for all angry,sad, surprise and more
        path = os.path.join(DataDirectory, category)
        class_num = classes.index(category)  # label 0,1
        for img in os.listdir(path):
            try:
                # list all images in each directory in the classes then
                # read it

                img_array = cv2.imread(os.path.join(path, img))
                img_size = 224  ## imageNet => 224 x 224
                new_array = cv2.resize(img_array, (img_size, img_size))

                training_data.append([new_array, class_num])
            except Exception as a:
                print(a)
                pass


create_training_data()
print(len(training_data))

random.shuffle(training_data)

x = []  ## data /feature
y = []  ## label

for features, label in training_data:
    x.append(features)
    y.append(label)

# Converting it to 4 dimesions
img_size = 224  ## imageNet => 224 x 224
x = np.array(x).reshape(-1, img_size, img_size, 3)

x = x.astype(np.float32) / 255.0  # normalizing it
y = np.array(y)

print("passed")

model = tensorflow.keras.applications.MobileNetV2()  ## pre trained model

base_input = model.layers[0].input  ## input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)  ## aading new layer after the output of global pulling layer
final_output = layers.Activation("relu")(final_output)  ## activation function
final_output = layers.Dense(64)(final_output)  ##
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(7, activation="softmax")(final_output)  ## my classes are 07,classification layer

new_model = keras.Model(inputs=base_input, outputs=final_output)
new_model.summary()
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
new_model.fit(x, y, epochs=15)  ## training
new_model.save("final_model_94p49.h5")

new_model = tf.keras.models.load_model("final_model_94p49.h5")

new_model.evaluate  ## test data, I will not test my test image

frame = cv2.imread("sad_boy.jpg")

frame.shape
## we need face detection algorithm
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
