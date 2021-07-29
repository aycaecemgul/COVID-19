from tensorflow.keras import models
import pickle
import numpy as np

CATEGORIES = ["0CAP","1COVID","2HEALTHY"]
IMG_SIZE=200


pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)


model_A= models.load_model("inceptionv3-covid-19.model")

model_B=models.load_model("vgg16covid-19.model")

model_C=models.load_model("vgg19-covid-19.model")


resultA = model_A.predict([X_test])

true=0
false=0
diger_hata=0
cov_hata=0
saglikli_hata=0


for i in range(len(y_test)):
    vote=np.argmax(resultA[i])
    if (vote== y_test[i]):
        true += 1
    else:
        false += 1
        if (y_test[i] == 0):
            diger_hata += 1
        elif (y_test[i] == 1):
            cov_hata += 1
        elif (y_test[i] == 2):
            saglikli_hata += 1

accuracy = true / len(y_test)
print("Inception v3")
print("accuracy= " + str(accuracy))
print("true= " + str(true))
print("false= " + str(false))
print("sağlıklı hata= " + str(saglikli_hata))
print("cov hata= " + str(cov_hata))
print("diğer hata= " + str(diger_hata))


resultB = model_B.predict([X_test])

true=0
false=0
diger_hata=0
cov_hata=0
saglikli_hata=0

for i in range(len(y_test)):
    vote=np.argmax(resultB[i])
    if ( vote== y_test[i]):
        true += 1
    else:
        false += 1
        if (y_test[i] == 0):
            diger_hata += 1
        elif (y_test[i] == 1):
            cov_hata += 1
        elif (y_test[i] == 2):
            saglikli_hata += 1

accuracy = true / len(y_test)
print("VGG-16")
print("accuracy= " + str(accuracy))
print("true= " + str(true))
print("false= " + str(false))
print("sağlıklı hata= " + str(saglikli_hata))
print("cov hata= " + str(cov_hata))
print("diğer hata= " + str(diger_hata))

resultC = model_C.predict([X_test])

true=0
false=0
diger_hata=0
cov_hata=0
saglikli_hata=0


for i in range(len(y_test)):
    vote=np.argmax(resultC[i])
    if ( vote == y_test[i]):
        true += 1
    else:
        false += 1
        if (y_test[i] == 0):
            diger_hata += 1
        elif (y_test[i] == 1):
            cov_hata += 1
        elif (y_test[i] == 2):
            saglikli_hata += 1

accuracy = true / len(y_test)
print("VGG-16")
print("accuracy= " + str(accuracy))
print("true= " + str(true))
print("false= " + str(false))
print("sağlıklı hata= " + str(saglikli_hata))
print("cov hata= " + str(cov_hata))
print("diğer hata= " + str(diger_hata))