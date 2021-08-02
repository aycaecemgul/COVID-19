import os
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
import skimage
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from PIL import Image
import pickle
import uuid
import pydicom as dicom
import gdcm
from PIL import Image
import skimage
from skimage.exposure import equalize_adapthist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = 200

DIR="COVID-CT-MD\\Other Cases"
NEW_DIR="COVID-CT-MD\\0CAP"
for file in os.listdir(DIR):
    file_path=os.path.join(DIR, file)
    for img in os.listdir(file_path):
        new_file_name=file+"-"+img
        img_path = os.path.join(file_path, img)
        img_array = asarray(Image.open(img_path).convert("RGB"))
        new_img_path =  os.path.join(NEW_DIR, new_file_name)
        plt.imsave(new_img_path, img_array, cmap="gray")

NEW_DIR="COVID-CT-MD\\1COVID"
DIR="COVID-CT-MD\\COVID-19 Cases"
for file in os.listdir(DIR):
    file_path=os.path.join(DIR, file)
    new_file_path = os.path.join(NEW_DIR, file)
    os.makedirs(new_file_path, exist_ok=True)
    for img in os.listdir(file_path):
        img_path = os.path.join(file_path, img)
        ds = dicom.read_file(img_path)
        img_array = ds.pixel_array
        new_img_path =  os.path.join(new_file_path, img)
        plt.imsave(new_img_path[:-3]+"png", img_array, cmap="gray")

NEW_DIR="COVID-CT-MD\\2HEALTHY"
DIR="COVID-CT-MD\\Normal Cases"
for file in os.listdir(DIR):
    file_path=os.path.join(DIR, file)
    new_file_path = os.path.join(NEW_DIR, file)
    os.makedirs(new_file_path, exist_ok=True)
    for img in os.listdir(file_path):
        img_path = os.path.join(file_path, img)
        ds = dicom.read_file(img_path)
        img_array = ds.pixel_array
        new_img_path =  os.path.join(new_file_path, img)
        plt.imsave(new_img_path[:-3]+"png", img_array, cmap="gray")


#FIXING IMAGE SIZE TO 200 X 200
def resize_dataset(DIR,IMG_SIZE):
    for category in os.listdir(DIR):
        DIR=os.path.join(DIR, category)
        for patient in os.listdir(DIR):
            path = os.path.join(DIR, patient)
            for img in os.listdir(path):
                img_path=os.path.join(path,img)
                img=asarray(Image.open(img_path).convert("RGB"))
                img=resize(img, (IMG_SIZE, IMG_SIZE))
                plt.imsave(img_path,img,cmap="gray")


DIR="COVID-CT-MD"
def clahe(DIR):
    for category in os.listdir(DIR):
        file_path=os.path.join(DIR, category)
        for img in os.listdir(file_path):
            img_path=os.path.join(file_path,img)
            img = asarray(Image.open(img_path))
            img=equalize_adapthist(img)
            img = equalize_adapthist(img)
            img_path=img_path[:-4]+"-clahe.png"
            plt.imsave(img_path, img, cmap="gray")


file_path="COVID-CT-MD\\0CAP"
for img in os.listdir(file_path):
    img_path=os.path.join(file_path,img)
    img_array = asarray(Image.open(img_path).convert("RGB"))
    img_array=equalize_adapthist(img_array)
    new_img_path=img[:-4]+"-clahe.png"
    new_img_path=os.path.join(file_path,new_img_path)
    plt.imsave(new_img_path, img_array, cmap="gray")
