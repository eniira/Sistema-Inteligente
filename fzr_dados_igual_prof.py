import os
from PIL import Image
import re

personagens_alvo = ("bart_simpson", "homer_simpson", "lisa_simpson", "marge_simpson", "maggie_simpson")
contagem: dict[str, int] = {}
if not os.path.exists("dataset_treated"):
    os.makedirs("dataset_treated")
for personagem in personagens_alvo:
    contagem[personagem] = 0
    if not os.path.exists(f"dataset_treated/{personagem}"):
        os.makedirs(f"dataset_treated/{personagem}")
    if not os.path.exists(f"dataset_treated/teste/{personagem}"):
        os.makedirs(f"dataset_treated/teste/{personagem}")


def copiarTreino(PATH: str):
    nome_personagem = ""
    try:
        nome_personagem = (re.match(r'[a-zA-Z]+', os.path.basename(os.path.normpath(PATH))).group(0))
        
    except AttributeError:
        print(f"arquivo inutil: {PATH}")
        return
    if PATH.endswith(".bmp"):
        achou = False
        for personagem in personagens_alvo:
            if nome_personagem in personagem:
                nome_personagem = personagem
                achou = True
        if not achou:
            # tem um personagem que não estamos trabalhando com aqui,
            # ou um arquivo bmp q n é de personagem
            return
        contagem[nome_personagem] += 1
        if not os.path.exists(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg"):
            Image.open(PATH).convert('RGB').save(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg")
        else:
            print(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg ja existe")
            

# criar uma pasta de validação e jogar as imgs la
def copiarValid(PATH: str):
    nome_personagem = ""
    try:
        nome_personagem = (re.match(r'[a-zA-Z]+', os.path.basename(os.path.normpath(PATH))).group(0))
        
    except AttributeError:
        print(f"arquivo inutil: {PATH}")
        return
    if PATH.endswith(".bmp"):
        achou = False
        for personagem in personagens_alvo:
            if nome_personagem in personagem:
                nome_personagem = personagem
                achou = True
        if not achou:
            # tem um personagem que não estamos trabalhando com aqui,
            # ou um arquivo bmp q n é de personagem
            return
        contagem[nome_personagem] += 1
        if not os.path.exists(f"dataset_treated/teste/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg"):
            Image.open(PATH).convert('RGB').save(f"dataset_treated/teste/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg")
        else:
            print(f"dataset_treated/teste/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg ja existe")
    
    pass

# e colocar as imagens com os indices finais como iniciais nela
def tratarPersonagem(PATH: str):
    nome_personagem = os.path.basename(os.path.dirname(PATH))
    if PATH.endswith(".jpg"):
        achou = False
        for personagem in personagens_alvo:
            if nome_personagem in personagem:
                nome_personagem = personagem
                achou = True
        if not achou:
            # tem um personagem que não estamos trabalhando com aqui,
            # ou um arquivo jpg q n é de personagem
            return
        contagem[nome_personagem] += 1
        #print(nome_personagem +"/", end='')
        #print(os.path.basename(PATH))
        if not os.path.exists(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg"):
            Image.open(PATH).convert('RGB').save(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg")
        else:
            print(f"dataset_treated/{nome_personagem}/pic_{contagem[nome_personagem]:04}.jpg já existe")
                
        
    
    

def percorrer_pastas(PATH: str):
# pega o nome da pasta (no caso do prof: Train/Valid, no do kaggle os nomes dos personagens_simpson)
    pasta = os.path.basename(os.path.normpath(PATH))
    funct = None
    # se a função foi chamada para uma pasta
    if os.path.basename(os.path.normpath(PATH)):
        print(f"foi chamada para a pasta {pasta}")
        if pasta == "Train": # treino professor
            funct = copiarTreino
            
        elif pasta == "Valid": # validacao professor
            funct = copiarValid
            
        elif pasta in personagens_alvo: # personagens kaggle
            funct = tratarPersonagem
            
        else:
            funct = percorrer_pastas
        
    # para cada arquivo na pasta
    if funct:
        for nome_arquivo in os.listdir(PATH):
            funct(PATH+"/"+nome_arquivo)
        
        
percorrer_pastas("Dataset_professor")
percorrer_pastas("simpsons_dataset")