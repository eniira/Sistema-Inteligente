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
import warnings

warnings.filterwarnings('ignore')

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


#RANDOM FOREST
#from sklearn.ensemble import KNeighborsClassifier
RF_model = SVC(kernel="linear")
#RF_model = KNeighborsClassifier()

# Train the model on training data
RF_model.fit(x_train, y_train) #For sklearn no one hot encoding

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(x_test)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_HGS = confusion_matrix(y_test, prediction_RF, labels=RF_model.classes_)
disp_HGS = ConfusionMatrixDisplay(confusion_matrix=cm_HGS,
                              display_labels=RF_model.classes_)
disp_HGS.plot()
plt.savefig('MC_NN.png')
plt.show()

# cm = confusion_matrix(y_test, prediction_RF)
# #print(cm)
# disp = sns.heatmap(cm, annot=True, cmap="icefire")

# disp.plot()
# plt.savefig('MC_NN.png')
# plt.show()

#Check results on a few select images
# n=np.random.randint(0, x_test.shape[0])
# img = x_test[n]
# plt.imshow(img)
# input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
# input_img_feature=VGG_model.predict(input_img)
# input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
# prediction_RF = RF_model.predict(input_img_features)[0] 
# prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
# print("The prediction for this image is: ", prediction_RF)
# print("The actual label for this image is: ", test_labels[n])