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

# t=20
# a=np.argmax(resultA[t])
# b=np.argmax(resultB[t])
# c=np.argmax(resultC[t])
#
# print(a)
# print(b)
# print(c)

# vote_list = [a, b, c]


def voting(vote_list):

    if ((vote_list[0] != vote_list[1]) and (vote_list[0] != vote_list[2]) and (vote_list[2] != vote_list[1])): #if ensemble cant decide vgg-19 will decide

        return vote_list[2]

    else:

        data = Counter(vote_list)
        return data.most_common(1)[0][0]


# print(voting(vote_list))
# print(y_test[20])

def ensemble_evaluation(y_test,resultA,resultB,resultC):
    evaluation_list=[]
    ensemble_list=[]
    true=0
    false=0
    for i in range(len(y_test)):
        vote_list=[resultA[i],resultB[i],resultC[i]]
        vote=voting(vote_list)
        ensemble_list.append(vote)
        if(vote==y_test[i]):
            evaluation_list.append(1)
            true+=1
        else:
            evaluation_list.append(0)
            false+=1
    accuracy= true/len(y_test)

    print("accuracy= "+ str(accuracy))
    print("true= "+str(true) )
    print("false= " + str(false))

ensemble_evaluation(y_test,resultA,resultB,resultC)