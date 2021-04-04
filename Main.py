import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras
from PIL import Image
from matplotlib import pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from numpy import asarray
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3
from numpy import asarray
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIR = 'C:\\Users\\aycae\\PycharmProjects\\Inceptionv3\\samples'
CATEGORIES = ["Covid","Healthy","Others"]
IMG_SIZE = 250

X=[]
y=[]


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

print("Pickle jar opened.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle= True)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.33, shuffle= True )

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


base_model = InceptionV3(
                                weights='imagenet',
                                include_top=False,
                                input_shape=(250, 250,3)
                                )

base_model.trainable=False

model = tf.keras.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.7):
      print("\nReached 70% val accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

aug = ImageDataGenerator()

BS=150
model.fit(aug.flow(X_train, y_train, batch_size = BS),validation_data = (X_val, y_val),epochs = 10,callbacks=[callbacks],verbose=2)



model.save("covid-19.model")