import os
import pickle
import random
import numpy as np
import skimage
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras
import cv2
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
import skimage
from skimage.color import rgb2gray,gray2rgb
from skimage.filters import median
from skimage import data
from skimage.morphology import disk
from skimage.filters import median,threshold_otsu
from skimage.exposure import equalize_adapthist
import PIL
from PIL import Image
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DIR = 'C:\\Users\\aycae\\PycharmProjects\\Inceptionv3\\samples'
# CATEGORIES = ["Covid","Healthy","Others"]
# training_data=[]
# IMG_SIZE = 250
# X=[]
# y=[]
#
# def image_preprocess(img):
#   img = rgb2gray(img)
#   med = median(img)
#   med= equalize_adapthist(med)
#   thresh = threshold_otsu(med)
#   img = med > thresh
#   img= resize(img, (250, 250))
#   img = gray2rgb(img)
#   return img
#
# for category in CATEGORIES:
#     path = os.path.join(DIR, category)
#
#     for img in os.listdir(path):
#         class_no = CATEGORIES.index(category)
#         filename = os.path.join(path, img)
#         img_array = asarray(Image.open(filename).convert('RGB'))
#         p_image = image_preprocess(img_array)
#         img_array = resize(img_array, (250, 250))
#
#         X.append(p_image)
#         y.append(class_no)
#         X.append(img_array)
#         y.append(class_no)
#
# X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# y = np.array(y)
# X=X/255.0
#
# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle= True)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.33, shuffle= True )


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

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

aug = ImageDataGenerator()
BS=150
model.fit(aug.flow(X_train, y_train, batch_size = BS),validation_data = (X_val, y_val),epochs = 10,verbose=2)
