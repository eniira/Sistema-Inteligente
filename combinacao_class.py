import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings('ignore')

x_treino = np.loadtxt('x_train.txt')
x_teste = np.loadtxt('x_test.txt')
y_treino = np.loadtxt('y_train.txt', dtype="str")
y_teste = np.loadtxt('y_test.txt', dtype="str")

SVMCLF = SVC(kernel='linear', random_state=42)

MLPCLF = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

KNNCLF = KNeighborsClassifier(n_neighbors=5)

classifierTypeNameInitial = [('SVM', SVMCLF), ('MLP', MLPCLF), ('KNN', KNNCLF)]

VotingCLF = VotingClassifier(estimators=classifierTypeNameInitial, voting='hard')

classifierTypeNameTotal = classifierTypeNameInitial + [('Voting', VotingCLF)]

classifierScore = {}

for nameCLF, CLF in classifierTypeNameTotal:
    CLF.fit(x_treino, y_treino)
    CLF_prediction = CLF.predict(x_teste)
    classifierScore[nameCLF] = metrics.f1_score(y_teste, CLF_prediction, average="weighted")

    if nameCLF == "Voting":
        print(classification_report(y_teste, CLF_prediction, target_names=CLF.classes_))

        matriz_confusao = confusion_matrix(y_teste, CLF_prediction, labels=CLF.classes_)
        display_matriz = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao,
                                    display_labels=CLF.classes_)

        fig, ax = plt.subplots(figsize=(10, 8))
        display_matriz.plot(ax=ax)
        plt.savefig('matriz_conf_combClass.png')
        plt.show()

print(classifierScore)