import numpy as np
import os
import matplotlib.pyplot as plt
from interfaz import metodo_de_la_potencia_def

# HIPER-PARAMETROS:
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
ITERACIONES = 10000
RANGO_AUTOVALORES = range(5,100)
DIFERENCIA_AUTOVALORES = 2

tolerancias = [[10**(-2), "blue"], [10**(-7), "orange"], [10**(-14), "red"]]
lista_autovalores = [[DIFERENCIA_AUTOVALORES*j for j in range(i, 0, -1)] for i in RANGO_AUTOVALORES]

for tolerancia in tolerancias:
    errores = []
    for autovalores in lista_autovalores:
        D = np.diag(autovalores)

        v = np.ones((D.shape[0], 1))

        v = v / np.linalg.norm(v)

        # Matriz de Householder
        B = np.eye(D.shape[0]) - 2 * (v @ v.T)

        # Matriz a diagonalizar, recordar B es simétrica y ortogonal
        M = B @ D @ B.T

        res = metodo_de_la_potencia_def(M, ITERACIONES, tolerancia[0])

        suma_errores = 0
        for i in range(len(M)):
            suma_errores = suma_errores + (float(np.linalg.norm(np.dot(M,res[:, i]) - (res[i,-2]*res[:, i]))))

        promedio_errores = suma_errores/len(M)
        errores.append(promedio_errores)

    plt.plot(RANGO_AUTOVALORES, errores, label=f"Tolerancia {tolerancia[0]}", color=tolerancia[1])

plt.xlabel('Tamaño de matriz')
plt.ylabel('Promedio de errores de autovalores y autovectores')
plt.yscale("log")
plt.title('Prueba método de la potencia')
plt.legend()
plt.savefig(os.path.join(CARPETA_IMAGENES, "prueba_metodo_potencia_error.png"))
plt.clf()

