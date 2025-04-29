import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from interfaz import metodo_de_la_potencia_def

# HIPER-PARAMETROS
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
Q = 1000


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

X_train = X[df["split"] == "train"]
X_train = X_train - X_train.mean(axis=0) # Centrar

X_cov = (X_train.T @ X_train)/(len(X_train) - 1)

autovalores = metodo_de_la_potencia_def(X_cov, 10000, 10**(-7))[:,-2]

varianza_total = sum(autovalores)
limite_varianza = varianza_total*0.95
p_optimo = 0

varianzas_acum = []
for p in range(len(autovalores)):
    var_acum = sum(autovalores[:p])
    varianzas_acum.append(var_acum)
    if var_acum >= limite_varianza and p_optimo == 0:
        p_optimo = p 


plt.plot(range(1, len(autovalores) + 1), varianzas_acum)
plt.axhline(limite_varianza, color="red", linestyle="dashed", label="95% varianza")
plt.axvline(p_optimo, color="blue", linestyle="dashed", label=f"p optimo = {p_optimo}")
plt.scatter(p_optimo, limite_varianza, color="black")
plt.xlabel('Componentes principales')
plt.ylabel('Varianza explicada')
plt.title('Pre-procesado con PCA')
plt.legend()

plt.savefig(os.path.join(CARPETA_IMAGENES, "varianzas_acumuladas.png"))
