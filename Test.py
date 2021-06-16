from tensorflow.keras import models
import pickle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray

CATEGORIES = ["0CAP","1COVID","2HEALTHY"]
IMG_SIZE=200


# pickle_in = open("X_test.pickle","rb")
# X_test = pickle.load(pickle_in)
#
# pickle_in = open("y_test.pickle","rb")
# y_test = pickle.load(pickle_in)


model= models.load_model("inceptionv3-covid-19.model")

test_path="D:\\Tez\\COVID-CT-MD-TEST\\0CAP\\cap001-IM0012-clahe.png"
test_image=asarray(Image.open(test_path).convert("RGB"))
test_image=np.array(test_image).reshape(-1,IMG_SIZE,IMG_SIZE,3)
print(test_image.shape)
predict=model.predict(test_image)

p=np.argmax(predict)
q=0

print("Actual label of the image: "+str(CATEGORIES[q]))
print("The prediction of the model: "+str(CATEGORIES[p]))

plt.imshow(test_image)

plt.show()
