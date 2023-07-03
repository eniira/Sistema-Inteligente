import numpy as np
import glob
import cv2

import os
from keras.applications.vgg16 import VGG16

novo_tamanho = 256  

x_treino = []
y_treino = []
x_teste = []
y_teste = [] 

caminho_dados_treinamento = "dataset_treated/*"
caminho_dados_teste = "dataset_treated/teste/*"

for caminho_diretorio in glob.glob(caminho_dados_treinamento):
    classe = caminho_diretorio.split("/")[-1]
    if classe == "teste":
      continue;
    for caminho_imagem in glob.glob(os.path.join(caminho_diretorio, "*.jpg")):
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)       
        imagem = cv2.resize(imagem, (novo_tamanho, novo_tamanho))
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        x_treino.append(imagem)
        y_treino.append(classe)

for caminho_diretorio in glob.glob(caminho_dados_teste):
    classe = caminho_diretorio.split("/")[-1]
    for caminho_imagem in glob.glob(os.path.join(caminho_diretorio, "*.jpg")):
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
        imagem = cv2.resize(imagem, (novo_tamanho, novo_tamanho))
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        x_teste.append(imagem)
        y_teste.append(classe)

x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
x_teste = np.array(x_teste)
y_teste = np.array(y_teste)

x_treino = x_treino / 255.0
x_teste = x_teste / 255.0

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(novo_tamanho, novo_tamanho, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary() 

x_treino_feat = VGG_model.predict(x_treino)
x_treino_formatado = x_treino_feat.reshape(x_treino_feat.shape[0], -1)

x_teste_feat = VGG_model.predict(x_teste)
x_teste_formatado = x_teste_feat.reshape(x_teste_feat.shape[0], -1)
 
np.savetxt('x_train.txt', x_treino_formatado)
np.savetxt('x_test.txt', x_teste_formatado)
np.savetxt('y_train.txt', y_treino)
np.savetxt('y_test.txt', y_teste, fmt='%s')
