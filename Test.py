from tensorflow.keras import models
import pickle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from collections import Counter

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
resultB = model_B.predict([X_test])
resultC = model_C.predict([X_test])

# file = open("inception_sonuc.txt", "w+")
# # Saving the array in a text file
# content = str(resultA)
# file.write(content)
# file.close()
#
# file = open("vgg16_sonuc.txt", "w+")
# # Saving the array in a text file
# content = str(resultB)
# file.write(content)
# file.close()

# file = open("vgg19_sonuc.txt", "w+")
# # Saving the array in a text file
# content = str(resultC)
# file.write(content)
# file.close()


def voting(resultA,resultB,resultC):

    sonuc_list=[]
    for i in range(3):
        sonuc=(resultA[i]+resultB[i]+resultC[i] )/3
        sonuc_list.append(sonuc)

    max_value = max(sonuc_list)
    vote = sonuc_list.index(max_value)

    return vote



def ensemble_evaluation(y_test,resultA,resultB,resultC):
    evaluation_list=[]
    ensemble_list=[]
    true=0
    false=0
    cov_hata=0
    saglikli_hata=0
    diger_hata=0
    for i in range(len(y_test)):

        vote=voting(resultA[i],resultB[i],resultC[i])
        ensemble_list.append(vote)
        if(vote==y_test[i]):
            evaluation_list.append(1)
            true+=1
        else:
            evaluation_list.append(0)
            false+=1
            if(y_test[i]==0):
                diger_hata+=1
            elif(y_test[i]==1):
                cov_hata+=1
            elif (y_test[i] == 2):
                saglikli_hata+=1

    accuracy= true/len(y_test)

    print("accuracy= "+ str(accuracy))
    print("true= "+str(true) )
    print("false= " + str(false))
    print("sağlıklı hata= "+ str(saglikli_hata))
    print("cov hata= "+ str(cov_hata))
    print("diğer hata= " + str(diger_hata))

ensemble_evaluation(y_test,resultA,resultB,resultC)

