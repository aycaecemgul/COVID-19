import pickle
import os
import numpy as np
from numpy import asarray
import skimage
from PIL import Image
import random
from matplotlib import pyplot as plt

CATEGORIES = ["0CAP","1COVID","2HEALTHY"]
DIR="D:\\Tez\\COVID-CT-MD-TEST\\"
X=[]
y=[]
training_data=[]
IMG_SIZE=200

for category in os.listdir(DIR):
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        img_path=os.path.join(path, img)
        class_no = CATEGORIES.index(category)
        try:
            img_array = asarray(Image.open(img_path).convert("RGB"))
            training_data.append([img_array, class_no])
        except Exception as e:
            print("error")

random.shuffle(training_data)

X=[]
y=[]

for painting,label in training_data:
    X.append(painting)
    y.append(label)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y)

print(X.shape)
print(y.shape)

pickle_out = open("X_test.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("Pickled.")