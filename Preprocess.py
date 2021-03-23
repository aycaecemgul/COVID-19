import os
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
import skimage
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from PIL import Image
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIR = 'C:\\Users\\aycae\\PycharmProjects\\Inceptionv3\\samples'
CATEGORIES = ["Covid","Healthy","Others"]
training_data=[]
IMG_SIZE = 250
X=[]
y=[]

def image_preprocess(img):

  return img

#FIXING IMAGE SIZE TO 250x250
for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        class_no = CATEGORIES.index(category)
        filename = os.path.join(path, img)
        img_array = asarray(Image.open(filename).convert('RGB'))
        img_array = resize(img_array, (250, 250))
        plt.imsave(filename,img_array)


for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        class_no = CATEGORIES.index(category)
        filename = os.path.join(path, img)
        img_array = asarray(Image.open(filename).convert('RGB'))
        p_image = skimage.exposure.equalize_adapthist(img_array) #CLAHE
        X.append(p_image)
        y.append(class_no)
        X.append(img_array)
        y.append(class_no)


X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y)
X=X/255.0

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()