import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pickle


CATEGORIES = ["Covid","Healthy","Others"]

model= models.load_model("covid-19.model")

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

predict=model.predict([X_test])

p=np.argmax(predict[1])
q=y_test[1]

print("Type: "+str(CATEGORIES[q]))
print("The prediction: "+str(CATEGORIES[p]))

