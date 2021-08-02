import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras
from PIL import Image
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from numpy import asarray
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from numpy import asarray
from skimage import exposure
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DIR = "curated_data\\curated_data"
CATEGORIES = ["1NonCOVID","2COVID","3CAP"]
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
    validation_split=0.2,
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
    validation_split=0.2,
    subset="validation"
)


#DATA AUGMENTATION
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

dataset_train = dataset_train.map(lambda x, y: (normalization_layer(x), y))

dataset_val = dataset_val.map(lambda x, y: (normalization_layer(x), y))

flip_layer=tf.keras.layers.experimental.preprocessing.RandomFlip(mode="vertical")

dataset_train = dataset_train.map(lambda x, y: (flip_layer(x), y))

dataset_val = dataset_val.map(lambda x, y: (flip_layer(x), y))

print("done")


model_name="vgg-19"
# base_model = InceptionV3(
#                                 weights='imagenet',
#                                 include_top=False,
#                                 input_shape=IMG_SHAPE
#                                 )
#
# base_model.trainable=True
#
# model = tf.keras.Sequential()
# model.add(base_model)
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(3, activation='softmax',kernel_regularizer='l1'))



# base_model = tf.keras.applications.vgg16.VGG16(
#                                 weights='imagenet',
#                                 include_top=False,
#                                 input_shape=IMG_SHAPE
#                                 )
#
# base_model.trainable=True
# model = tf.keras.Sequential()
# model.add(base_model)
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(3, activation='softmax',kernel_regularizer='l1'))


base_model = tf.keras.applications.vgg19.VGG19(
                                weights='imagenet',
                                include_top=False,
                                input_shape=IMG_SHAPE
                                )

base_model.trainable=True
model = tf.keras.Sequential()
model.add(base_model)
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(3, activation='softmax',kernel_regularizer='l1'))


model.summary()




model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


filepath = model_name+"-covid-19.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)


history=model.fit(dataset_train,validation_data = dataset_val,batch_size=32,epochs = 8,callbacks=[checkpoint],verbose=2,shuffle=True)

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


model.save(model_name+"covid-19.model")

# scores = model.evaluate(X_test, y_test, verbose=0)
# print("%s: %.2f" % (model.metrics_names[0], scores[0]))
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# t=27
#
# CATEGORIES = ["0CAP","1COVID","2HEALTHY"]
# print("The prediction is:")
# p=np.argmax(predict[t])
#
# q=y_test[t]
# print("Actual label of the image: "+str(CATEGORIES[q]))
# print("The prediction of the model: "+str(CATEGORIES[p]))

# from keras.models import load_model
# saved_model = load_model('best_model.h5')

# test_acc = saved_model.evaluate(testX, testy, verbose=0)

