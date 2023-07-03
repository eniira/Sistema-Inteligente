import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

SIZE = 256  #Resize images

train_labels = [] 

for directory_path in glob.glob("dataset_treated/*"):
    label = directory_path.split("/")[-1]
    if label == "teste":
      continue;
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        train_labels.append(label)

x_train = np.loadtxt('x_train.txt')
x_test = np.loadtxt('x_test.txt')
y_train = np.array(train_labels)
y_test = np.loadtxt('y_test.txt', dtype="str")


# split the data set into training and test data sets
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtotal, Ytotal, test_size=0.5)

# create a standard scaler
# scaler = StandardScaler()
# scale the training and test input data
# fit_transform performs both fit and transform at the same time
# XtrainScaled = scaler.fit_transform(Xtrain)
# here we only need to transform
# XtestScaled = scaler.transform(Xtest)

# Normalizar os dados de entrada
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ===============================
# Instanciando os Classificadores

# Support Vector Machine
SVMCLF = SVC(kernel='rbf', random_state=42)

# Multi-Layer Perceptron
MLPCLF = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# k-Nearest Neighbors
KNNCLF = KNeighborsClassifier(n_neighbors=5)

# ==============================================
# Criando uma lista de tuplas de classificadores
# essa lista é usada pelo VotingClassifier
# (classifier_name, classifier_object)
classifierTypeNameInitial = [('SVM', SVMCLF), ('MLP', MLPCLF), ('KNN', KNNCLF)]

# criando o classificador voting (combinação dos outros classificadores)
VotingCLF = VotingClassifier(estimators=classifierTypeNameInitial, voting='hard')

# criando a lista final de tuplas de classificadores
classifierTypeNameTotal = classifierTypeNameInitial + [('Voting', VotingCLF)]

# armazena a pontuação dos classificadores
classifierScore = {}

# iterando sobre os classificadores, treinando-os, testando-os e calculando acurácia
for nameCLF, CLF in classifierTypeNameTotal:
    CLF.fit(x_train_scaled, y_train)
    CLF_prediction = CLF.predict(x_test_scaled)
    classifierScore[nameCLF] = accuracy_score(y_test, CLF_prediction)

print(classifierScore)

