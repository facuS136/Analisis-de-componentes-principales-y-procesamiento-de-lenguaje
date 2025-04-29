import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from implementaciones.knn import knn
from interfaz import metodo_de_la_potencia_def

# HIPER-PARAMETROS
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
Q = 1000
RANGO_K = range(5, 51, 1)
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

# Datos de ENTRENAMIENTO
X_train = X[df["split"] == "train"]

# Indices peliculas ENTRENAMIENTO
df_train = df[df["split"] == "train"].reset_index(drop=True)

# -- Guardamos los datos de cada genero para crear subconjutos de igual proporcion luego--
crime_indices = df_train[df_train["Genre"] == "crime"].index
romance_indices = df_train[df_train["Genre"] == "romance"].index
science_fiction_indices = df_train[df_train["Genre"] == "science fiction"].index
western_indices = df_train[df_train["Genre"] == "western"].index

# definimos el rango de P como solamente la cantidad de filas de cada particion de entrenamiento, que seria la siguiente
RANGO_P = range(5, 3*(X_train.shape[0]//4) + 1, 5)

# guardamos los mejores k y p que vamos encontrando
best_k = RANGO_K.start
best_p = RANGO_P.start

# estas listas son para hacer los graficos
lista_k = []
lista_p = []
lista_performance = []

best_promedio = 0

folds = list()
Vs = list()

# precomputo todos los folds y su Vs
for i in [0, 1, 2, 3]:  # itero cada una de los subconjuntos posibles de peliculas
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

    # Center train and test data
    mean = train_data.mean(axis=0)
    train_data = train_data - mean
    test_data = test_data - mean

    # generos peliculas ENTRENAMIENTO
    train_types = df_train.loc[train_indices, "Genre"].values
    # generos peliculas DESARROLLO
    test_types = df_train.loc[test_indices, "Genre"].values

    # Obtenemos la matriz de covarianza
    train_data_cov = (train_data.T@train_data)/(len(train_data) - 1)

    # Asserts para verificar las particiones
    assert len(train_indices) == 240, f"Error: train_indices tiene {len(train_indices)}" 
    assert len(test_indices) == 80, f"Error: test_indices tiene {len(test_indices)}"
    assert all(df_train.loc[train_indices, "split"] == "train"), "Error en train_indices"
    assert all(df_train.loc[crime_indices, "Genre"] == "crime"), "Error: en crime_indices"
    assert all(df_train.loc[romance_indices, "Genre"] == "romance"), "Error: en romance_indices"
    assert all(df_train.loc[science_fiction_indices, "Genre"] == "science fiction"), "Error: en science_fiction_indices"
    assert all(df_train.loc[western_indices, "Genre"] == "western"), "Error: en western_indices"

    V = metodo_de_la_potencia_def(train_data_cov, 10000, 10**(-7))[:,:-2]

    folds.append((train_data, test_data, train_types, test_types))
    Vs.append(V)
    


for p in RANGO_P:
    # Hacemos el cambio de dimension de X haciendo X.V, pero tomando las primeras p columnas de V
    for k in RANGO_K:
        # Hago el promedio del performance entre todas las particiones
        promedio = 0
        for i in [0, 1, 2, 3]:  # itero cada una de los subconjuntos posibles de peliculas
            # tokens peliculas ENTRENAMIENTO
            train_data = folds[i][0]
            # tokens peliculas DESARROLLO
            test_data = folds[i][1]

            # generos peliculas ENTRENAMIENTO
            train_types = folds[i][2]
            # generos peliculas DESARROLLO
            test_types = folds[i][3]

            # autovalores del fold
            V = Vs[i]

            train_data_resized = train_data @ (V[:,:p])
            test_data_resized = test_data @ (V[:,:p])
 
            aciertos = knn(k, train_data_resized, test_data_resized, train_types, test_types)
            porcentaje_aciertos = (sum(aciertos)*100.0)/len(aciertos)
            promedio += porcentaje_aciertos
        
        promedio = promedio / 4

        lista_p.append(p)
        lista_k.append(k)
        lista_performance.append(promedio)

        if promedio > best_promedio:
            best_promedio = promedio
            best_k = k
            best_p = p
        print(f"Performance para k = {k} y p = {p} : {promedio}%")

print(f"El mejor promedio para Q = {Q} fue {best_promedio}% para k = {best_k} y p = {best_p}")

# Hacemos un grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.array(lista_p)
y = np.array(lista_k)
z = np.array(lista_performance)

ax.scatter(x, y, z, c=z, cmap='viridis')

ax.set_xlabel('p componentes')
ax.set_ylabel('k vecinos')
ax.set_zlabel('Performance')
ax.set_title('Grafico Pipeline Final')
plt.savefig(os.path.join(CARPETA_IMAGENES,"pipeline_final.png"))
plt.show()
