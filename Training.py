import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from numpy import asarray
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3
from numpy import asarray

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DIR = 'D:\Tez\COVID-CT-MD'
CATEGORIES = ["0CAP","1COVID","2HEALTHY"]
IMG_SIZE = 200
IMG_SHAPE=(200,200,3)

dataset_train=tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    labels="inferred",
    label_mode="int",
    class_names=CATEGORIES,
    color_mode="rgb",
    image_size=(IMG_SIZE,IMG_SIZE),
    shuffle=True,
    seed=123,
    validation_split=0.33,
    subset="training"
)

dataset_val=tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    labels="inferred",
    label_mode="int",
    class_names=CATEGORIES,
    color_mode="rgb",
    image_size=(IMG_SIZE,IMG_SIZE),
    shuffle=True,
    seed=123,
    validation_split=0.33,
    subset="validation"
)

# Found 24802 files belonging to 3 classes.
# Using 16618 files for training.
# Found 24802 files belonging to 3 classes.
# Using 8184 files for validation.


#DATA AUGMENTATION
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

dataset_train = dataset_train.map(lambda x, y: (normalization_layer(x), y))

dataset_val = dataset_val.map(lambda x, y: (normalization_layer(x), y))

flip_layer=tf.keras.layers.experimental.preprocessing.RandomFlip(mode="vertical")

dataset_train = dataset_train.map(lambda x, y: (flip_layer(x), y))

dataset_val = dataset_val.map(lambda x, y: (flip_layer(x), y))



base_model = InceptionV3(
                                weights='imagenet',
                                include_top=False,
                                input_shape=IMG_SHAPE
                                )

base_model.trainable=True

model = tf.keras.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


filepath="inceptionv3_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


history=model.fit(dataset_train,batch_size=32,validation_data = dataset_val,epochs = 10,verbose=2,callbacks=callbacks_list,shuffle=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


model.save("inceptionv3-covid-19.model")