import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings('ignore')

x_treino = np.loadtxt('x_train.txt')
x_teste = np.loadtxt('x_test.txt')
y_treino = np.loadtxt('y_train.txt', dtype="str")
y_teste = np.loadtxt('y_test.txt', dtype="str")

modelo = SVC(kernel="linear")

modelo.fit(x_treino, y_treino) 

y_predito = modelo.predict(x_teste)

print(classification_report(y_teste, y_predito, target_names=modelo.classes_))

matriz_confusao = confusion_matrix(y_teste, y_predito, labels=modelo.classes_)
display_matriz = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao,
                              display_labels=modelo.classes_)

fig, ax = plt.subplots(figsize=(10, 8))
display_matriz.plot(ax=ax)
#plt.savefig('matriz_conf_kNN.png')
plt.show()