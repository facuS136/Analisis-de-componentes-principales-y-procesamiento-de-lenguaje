import numpy as np
from interfaz import metodo_de_la_potencia_def

A = np.array([[7, 2, 3], [0, 2, 0], [-6, -2, -2]], dtype=float)
result = metodo_de_la_potencia_def(A, 1000, 0.0)
print(f"Resultado obtenido: {result[:,-1]}")
print("Resusltado esperado: 4, 2, 1")