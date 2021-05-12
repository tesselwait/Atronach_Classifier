import os, shutil
import tensorflow as tf
import keras
keras.__version__
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'data set directory' #path to image files
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_fire_dir = os.path.join(train_dir, 'fire')
train_frost_dir = os.path.join(train_dir, 'frost')
train_storm_dir = os.path.join(train_dir, 'storm')
train_none_dir = os.path.join(train_dir, 'none')
validation_fire_dir = os.path.join(validation_dir, 'fire')
validation_frost_dir = os.path.join(validation_dir, 'frost')
validation_storm_dir = os.path.join(validation_dir, 'storm')
validation_none_dir = os.path.join(validation_dir, 'none')
test_fire_dir = os.path.join(test_dir, 'fire')
test_frost_dir = os.path.join(test_dir, 'frost')
test_storm_dir = os.path.join(test_dir, 'storm')
test_none_dir = os.path.join(test_dir, 'none')

print('total training fire images:', len(os.listdir(train_fire_dir)))
print('total training frost images:', len(os.listdir(train_frost_dir)))
print('total training storm images:', len(os.listdir(train_storm_dir)))
print('total training none images:', len(os.listdir(train_none_dir)))
print('total validation fire images:', len(os.listdir(validation_fire_dir)))
print('total validation frost images:', len(os.listdir(validation_frost_dir)))
print('total validation storm images:', len(os.listdir(validation_storm_dir)))
print('total validation none images:', len(os.listdir(validation_none_dir)))
print('total test fire images:', len(os.listdir(test_fire_dir)))
print('total test frost images:', len(os.listdir(test_frost_dir)))
print('total test storm images:', len(os.listdir(test_storm_dir)))
print('total test none images:', len(os.listdir(test_none_dir)))

datagen = ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

fnames = [os.path.join(train_fire_dir, fname) for fname in os.listdir(train_fire_dir)]

img_path = fnames[80]

img = image.load_img(img_path, target_size=(360, 640))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['acc'],
              optimizer='adam'
              )

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(360, 640),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(360, 640),
        batch_size=20,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=40,
      epochs=175,
      validation_data=validation_generator,
      validation_steps=20)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(360, 640),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=20)
print('test acc:', test_acc)

model.save('atronach_model_68.h5')

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
