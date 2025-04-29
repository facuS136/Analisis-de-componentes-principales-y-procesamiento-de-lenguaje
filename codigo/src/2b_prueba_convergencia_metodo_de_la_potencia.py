import numpy as np
import os
import matplotlib.pyplot as plt
from interfaz import metodo_de_la_potencia_def

# HIPER-PARAMETROS:
CARPETA_IMAGENES = os.path.join(os.path.dirname(__file__), 'graficos')
REPETICIONES_PROMEDIO = 100
EPSILONS = np.logspace(-4, 0, num=100)

def graficar(x, ys, y_errs, nom_img, titulo, nom_x, nom_y, legends):
    colores = ['r', 'g', 'b', 'm', 'y']
    plt.figure()
    for i in range(ys.shape[0]):
        plt.errorbar(x, ys[i], yerr=y_errs[i], color=colores[i], label=legends[i])
    plt.xlabel(nom_x)
    plt.ylabel(nom_y)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(titulo)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(CARPETA_IMAGENES, f"{nom_img}.png"))

iteraciones_autovalor = np.zeros((5, len(EPSILONS)), dtype=float)
errores_autovalor = np.zeros((5, len(EPSILONS)), dtype=float)
desvios_iteraciones = np.zeros((5, len(EPSILONS)), dtype=float)
desvios_errores = np.zeros((5, len(EPSILONS)), dtype=float)

index_e = 0

for e in EPSILONS:
    D = np.diag([10.0, 10.0 - e, 5.0, 2.0, 1.0])

    errores = np.zeros((len(D), REPETICIONES_PROMEDIO), dtype=float)
    iteraciones = np.zeros((len(D), REPETICIONES_PROMEDIO), dtype=float)

    # Hacemos un promedio
    for rep in range(REPETICIONES_PROMEDIO):

        v = np.random.rand(len(D), 1)   # Creo un v aleatorio

        v = v / np.linalg.norm(v)       # Lo normalizo

        # Matriz de Householder
        B = np.eye(D.shape[0]) - 2 * (v @ v.T)

        # Matriz a diagonalizar, recordar B es simétrica y ortogonal
        M = B @ D @ B.T

        res = metodo_de_la_potencia_def(M, 999999, 10**(-7))    # ponemos como iteraciones un numero muy alto para que corte por la tolerancia

        for i in range(len(M)):
            # guardo y sumo el error de cada autovalor y autovector
            errores[i, rep] = float(np.linalg.norm(np.dot(M, res[:, i]) - (res[i, -2] * res[:, i])))
            # guardo y sumo las iteraciones para calcular cada autovalor/autovector
            iteraciones[i, rep] = res[i, -1]

    errores_promedio = np.mean(errores, axis=1)
    iteraciones_promedio = np.mean(iteraciones, axis=1)
    errores_std = np.std(errores, axis=1)
    iteraciones_std = np.std(iteraciones, axis=1)

    errores_autovalor[:, index_e] = errores_promedio
    iteraciones_autovalor[:, index_e] = iteraciones_promedio
    desvios_errores[:, index_e] = errores_std
    desvios_iteraciones[:, index_e] = iteraciones_std

    index_e += 1

graficar(EPSILONS, errores_autovalor, desvios_errores, "grafico_errores", "Grafico errores", "epsilon", "error", ["error v_1 = 10", "error v_2 = 10 - e", "error v_3 = 5", "error v_4 = 2", "error v_5 = 1"])
graficar(EPSILONS, iteraciones_autovalor, desvios_iteraciones, "grafico_iteraciones", "Grafico iteraciones", "epsilon", "iteracion", ["iteraciones v_1 = 10", "iteraciones v_2 = 10 - e", "iteraciones v_3 = 5", "iteraciones v_4 = 2", "iteraciones v_5 = 1"])
