import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

DataDirectory = "archive/test/"
classes = ["0", "1", "2", "3", "4", "5", "6"]

# Define data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Define batch size
batch_size = 16

# Define train and validation generators
train_generator = datagen.flow_from_directory(
    DataDirectory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training')

validation_generator = datagen.flow_from_directory(
    DataDirectory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation')

# Define model architecture
model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output
final_output = layers.Dense(128)(base_output)
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(7, activation="softmax")(final_output)
new_model = keras.Model(inputs=base_input, outputs=final_output)

# Compile the model
new_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

# Train the model using the generators
history = new_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Save the trained model
new_model.save("final_model_94p49.h5")
