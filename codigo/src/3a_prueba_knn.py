import numpy as np
import pandas as pd
import os
from implementaciones.knn import knn
import matplotlib.pyplot as plt

# HIPER-PARAMETROS
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
RANGO_Q = [500, 1000, 5000]
k = 5

lista_aciertos = list()

for Q in RANGO_Q:

    # Pre-procesado de datos
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


    # Hacemos KNN para cada K en el rango RANGO_K
    # Indices peliculas ENTRENAMIENTO
    train_indices = df[df["split"] == "train"].index
    # Indices peliculas DESARROLLO
    test_indices = df[df["split"] == "test"].index

    # Datos de ENTRENAMIENTO
    train_data = X[train_indices]
    # Datos de DESARROLLO
    test_data = X[test_indices]
    # Guardamos los generos de cada dato
    train_types = df.loc[train_indices, "Genre"].values
    test_types = df.loc[test_indices, "Genre"].values

    aciertos = knn(k, train_data, test_data, train_types, test_types)
    porcentaje_aciertos = (sum(aciertos)*100.0)/len(aciertos)
    print(f"Aciertos con Q = {Q} y k = {k} : {sum(aciertos)} de {len(aciertos)} (%{porcentaje_aciertos})")
    lista_aciertos.append(porcentaje_aciertos)

print(lista_aciertos)
print(RANGO_Q)

plt.bar(list(range(len(RANGO_Q))), lista_aciertos, width=0.4)
plt.xticks(list(range(len(RANGO_Q))), RANGO_Q)
plt.xlabel('Q componentes')
plt.ylabel('Porcentaje de aciertos')
plt.ylim(0, 100)
plt.title(f'Grafico porcentaje de aciertos con KNN para k = {k} con distinto Q')
plt.savefig(os.path.join(CARPETA_IMAGENES, "clasificador_k_5.png"))
