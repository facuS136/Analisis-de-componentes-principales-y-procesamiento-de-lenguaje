import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from implementaciones.knn import knn
from interfaz import metodo_de_la_potencia_def

# HIPER-PARAMETROS
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
Q = 1000
K = 11
P = 30
#----------------------------------------------#

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'wiki_movie_plots_deduped_sample.csv')
df = pd.read_csv(csv_path)

tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)

unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))
X = np.zeros((len(df), len(unique_tokens)), dtype=int)
for i, row in df.iterrows():
    for token in row["tokens"].split():
        if unique_tokens_dict.get(token,False)!=False:
            X[i, unique_tokens_dict[token]] += 1

# Datos de TRAIN
X_train = X[df["split"] == "train"]
# Datos de TEST
X_test = X[df["split"] == "test"]
# Guardamos los generos de cada dato
train_types = df.loc[df["split"] == "train", "Genre"].values
test_types = df.loc[df["split"] == "test", "Genre"].values

# Center train and test data
mean = X_train.mean(axis=0)
X_train = X_train - mean
X_test = X_test - mean

# calculamos la MATRIZ DE COVARIANZA
X_train_cov = (X_train.T@X_train)/(len(X_train) - 1)

V = metodo_de_la_potencia_def(X_train_cov, 10000, 10**(-7))[:,:-2]

# Hago cambio de BASE
X_train_resized = X_train @ (V[:,:P])
X_test_resized = X_test @ (V[:,:P])

aciertos = knn(K, X_train_resized, X_test_resized, train_types, test_types)
porcentaje_aciertos = (sum(aciertos)*100.0)/len(aciertos)
print(f"Aciertos con Q = {Q} y k = {K} : {sum(aciertos)} de {len(aciertos)} (%{porcentaje_aciertos})")
