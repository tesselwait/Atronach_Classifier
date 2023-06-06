import os, shutil
import tensorflow as tf
import keras
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
import tensorflow.keras.utils
from keras import activations
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D

base_dir = '**DATASET PATH**' #path to image files

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_fire_dir = os.path.join(train_dir, 'fire')
train_frost_dir = os.path.join(train_dir, 'frost')
train_storm_dir = os.path.join(train_dir, 'storm')
train_mudcrab_dir = os.path.join(train_dir, 'mudcrab')
train_bandit_dir = os.path.join(train_dir, 'bandit')
train_bear_dir = os.path.join(train_dir, 'bear')
train_spriggan_dir = os.path.join(train_dir, 'spriggan')
train_dremora_dir = os.path.join(train_dir, 'dremora')
train_fox_dir = os.path.join(train_dir, 'fox')
train_troll_dir = os.path.join(train_dir, 'troll')
validation_fire_dir = os.path.join(validation_dir, 'fire')
validation_frost_dir = os.path.join(validation_dir, 'frost')
validation_storm_dir = os.path.join(validation_dir, 'storm')
validation_mudcrab_dir = os.path.join(validation_dir, 'mudcrab')
validation_bandit_dir = os.path.join(validation_dir, 'bandit')
validation_bear_dir = os.path.join(validation_dir, 'bear')
validation_spriggan_dir = os.path.join(validation_dir, 'spriggan')
validation_dremora_dir = os.path.join(validation_dir, 'dremora')
validation_fox_dir = os.path.join(validation_dir, 'fox')
validation_troll_dir = os.path.join(validation_dir, 'troll')

test_fire_dir = os.path.join(test_dir, 'fire')
test_frost_dir = os.path.join(test_dir, 'frost')
test_storm_dir = os.path.join(test_dir, 'storm')
test_mudcrab_dir = os.path.join(test_dir, 'mudcrab')
test_bandit_dir = os.path.join(test_dir, 'bandit')
test_bear_dir = os.path.join(test_dir, 'bear')
test_spriggan_dir = os.path.join(test_dir, 'spriggan')
test_dremora_dir = os.path.join(test_dir, 'dremora')
test_fox_dir = os.path.join(test_dir, 'fox')
test_troll_dir = os.path.join(test_dir, 'troll')

print('total training fire images:', len(os.listdir(train_fire_dir)))
print('total training frost images:', len(os.listdir(train_frost_dir)))
print('total training storm images:', len(os.listdir(train_storm_dir)))
print('total training mudcrab images:', len(os.listdir(train_mudcrab_dir)))
print('total training bandit images:', len(os.listdir(train_bandit_dir)))
print('total training bear images:', len(os.listdir(train_bear_dir)))
print('total training spriggan images:', len(os.listdir(train_spriggan_dir)))
print('total training dremora images:', len(os.listdir(train_dremora_dir)))
print('total training fox images:', len(os.listdir(train_fox_dir)))
print('total training troll images:', len(os.listdir(train_troll_dir)))
print('total validation fire images:', len(os.listdir(validation_fire_dir)))
print('total validation frost images:', len(os.listdir(validation_frost_dir)))
print('total validation storm images:', len(os.listdir(validation_storm_dir)))
print('total validation mudcrab images:', len(os.listdir(validation_mudcrab_dir)))
print('total validation bandit images:', len(os.listdir(validation_bandit_dir)))
print('total validation bear images:', len(os.listdir(validation_bear_dir)))
print('total validation spriggan images:', len(os.listdir(validation_spriggan_dir)))
print('total validation dremora images:', len(os.listdir(validation_dremora_dir)))
print('total validation fox images:', len(os.listdir(validation_fox_dir)))
print('total validation troll images:', len(os.listdir(validation_troll_dir)))
print('total test fire images:', len(os.listdir(test_fire_dir)))
print('total test frost images:', len(os.listdir(test_frost_dir)))
print('total test storm images:', len(os.listdir(test_storm_dir)))
print('total test mudcrab images:', len(os.listdir(test_mudcrab_dir)))
print('total test bandit images:', len(os.listdir(test_bandit_dir)))
print('total test bear images:', len(os.listdir(test_bear_dir)))
print('total test spriggan images:', len(os.listdir(test_spriggan_dir)))
print('total test dremora images:', len(os.listdir(test_dremora_dir)))
print('total test fox images:', len(os.listdir(test_fox_dir)))
print('total test troll images:', len(os.listdir(test_troll_dir)))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['acc'],
              optimizer='adam'
              )
filepath="checkpoint\\weights.best-3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=callbacks_list)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

model.save('skyrim_forest_model_3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
