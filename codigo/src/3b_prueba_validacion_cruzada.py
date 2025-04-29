import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from implementaciones.knn import knn

# HIPER-PARAMETROS
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
RANGO_K = range(1, 100)

for Q in [500, 1000, 5000]:
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
    
    # Datos de ENTRENAMIENTO
    X_train = X[df["split"] == "train"]

    # Indices peliculas ENTRENAMIENTO
    df_train = df[df["split"] == "train"].reset_index(drop=True)

    # -- CROSS VALIDATION --
    crime_indices = df_train[df_train["Genre"] == "crime"].index
    romance_indices = df_train[df_train["Genre"] == "romance"].index
    science_fiction_indices = df_train[df_train["Genre"] == "science fiction"].index
    western_indices = df_train[df_train["Genre"] == "western"].index

    best_k = RANGO_K.start
    best_promedio = 0
    promedios = list()
    for k in RANGO_K:
        promedio_actual = 0
        for i in [0, 1, 2, 3]:  # itero cada una de los subconjuntos posibles de peliculas
            # Indices ENTRENAMIENTO
            train_indices = pd.Index(
            list(crime_indices[:20*i]) + list(crime_indices[20*(i+1):]) +
            list(romance_indices[:20*i]) + list(romance_indices[20*(i+1):]) +
            list(science_fiction_indices[:20*i]) + list(science_fiction_indices[20*(i+1):]) +
            list(western_indices[:20*i]) + list(western_indices[20*(i+1):])
            )
            # Indices DESARROLLO
            test_indices = pd.Index(
            list(crime_indices[20*i:20*(i+1)]) +
            list(romance_indices[20*i:20*(i+1)]) +
            list(science_fiction_indices[20*i:20*(i+1)]) +
            list(western_indices[20*i:20*(i+1)])
            )
            # tokens peliculas ENTRENAMIENTO
            train_data = X_train[train_indices]
            # tokens peliculas DESARROLLO
            test_data = X_train[test_indices]
            # generos peliculas ENTRENAMIENTO
            train_types = df_train.loc[train_indices, "Genre"].values
            # generos peliculas DESARROLLO
            test_types = df_train.loc[test_indices, "Genre"].values
            
            aciertos = knn(k, train_data, test_data, train_types, test_types)
            porcentaje_aciertos = (sum(aciertos)*100.0)/len(aciertos)
            promedio_actual += porcentaje_aciertos

            print(f"Aciertos k = {k} : {sum(aciertos)} de {len(aciertos)} (%{porcentaje_aciertos})")
        
        promedio_actual = promedio_actual / 4
        promedios.append(promedio_actual)
        if promedio_actual > best_promedio:
            best_promedio = promedio_actual
            best_k = k
        

        print(f"Performance para k = {k} : {promedio_actual}%")

    print(f"El mejor promedio para Q = {Q} fue {best_promedio}% para k = {best_k}")

    
    plt.plot(list(RANGO_K), promedios, label="promedio eficiencia")
    plt.axhline(best_promedio, color="red", linestyle="dashed", label=f"Mejor promedio (%{best_promedio})")
    plt.axvline(best_k, color="blue", linestyle="dashed", label=f"Mejor k ({best_k})")
    plt.scatter(best_k, best_promedio, color="black")
    plt.xlabel('k')
    plt.ylabel('promedio eficiencia')
    plt.title(f'4-fold cross-validation para Q = {Q}')
    plt.legend()
    plt.savefig(os.path.join(CARPETA_IMAGENES, f"4FoldCrossValidationQ{Q}.png"))
    plt.clf()
