import os, shutil
import tensorflow as tf
import keras
keras.__version__
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import models
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'DATASET PATH' #path to image files
test_dir = os.path.join(base_dir, 'test')

test_fire_dir = os.path.join(test_dir, 'fire')
test_frost_dir = os.path.join(test_dir, 'frost')
test_storm_dir = os.path.join(test_dir, 'storm')
test_none_dir = os.path.join(test_dir, 'none')

print('total test fire images:', len(os.listdir(test_fire_dir)))
print('total test frost images:', len(os.listdir(test_frost_dir)))
print('total test storm images:', len(os.listdir(test_storm_dir)))
print('total test none images:', len(os.listdir(test_none_dir)))


model = load_model('MODEL_FILENAME')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(480, 854),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=20)
print('test acc:', test_acc)
