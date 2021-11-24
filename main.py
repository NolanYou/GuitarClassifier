import os
from pathlib import Path

import random

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import matplotlib as plt

import util
from keras_model import keras_guitar_model

BATCH_SIZE = 128
num_classes = 3
epochs = 15

#sampling rate of the wav file
SAMPLING_RATE = 16000
#path to the dataset
DATASET_AUDIO_PATH = ""
#generate the random shuffle seed
SHUFFLE_SEED = random.randint(3, 9)

# Percentage of samples to use for validation
VALID_SPLIT = 0.5



#input image dimensions TODO: tweak
img_rows, img_cols = 28, 28

x_train = False #TODO: implement
y_train = False
x_test = False
y_test = False

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

x_train = False #x_train.reshape(numimages, img_rows, img_cols, channels)
y_train = False #y_train.reshape(numimages, img_rows, img_cols, channels)
#note to self: the image size, channels of both should agree with each other
#y channels are labels

#splice my data it into 45 segements
#then, use numpy to shuffle it.
#take first half, use as training
#take second half, use as testing
#make it array of label classes or whatever, each class has 2 properties: label, whatever.
#generate needed arrays when done.




# Get the list of audio file paths along with their corresponding labels

class_names = os.listdir(DATASET_AUDIO_PATH)
print("Our class names: {}".format(class_names,))

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name,))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)


print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)





# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation
train_ds = util.paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = util.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

#This takes the audio snippet, ffts it to make it useable
train_ds = train_ds.map(
    lambda x, y: (util.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.map(
    lambda x, y: (util.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)


model = keras_guitar_model(class_names,train_ds,valid_ds)