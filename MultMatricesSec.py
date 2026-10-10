import numpy as np
import multiprocessing as mp

# Prueba secuncial
def multiplicacion_secuencial(a, b):
    c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(b[0])):
            suma = 0

            for k in range(len(a[0])):
                suma += a[i][k] * b[k][j]

            c[i][j] = suma

    return c

# Multiplicación en paralelo. Este coso devuelve el resultado de pasar una fila de a por la matriz b
# y pus es lo q se paraleliza pq cada fila se ejecuta de manera independiente
def calcular_fila(fila_a, b):
    fila_resultado = [0 for _ in range(len(b[0]))]
    
    for j in range(len(b[0])):
        suma = 0
        for k in range(len(fila_a)):
            suma += fila_a[k] * b[k][j]
        fila_resultado[j] = suma
        
    return fila_resultado

def multiplicacion_paralela(a, b):
    # Esto se va trabajar de la sig manera: cada fila tiene q ser multiplicada por la matriz b, tons
    # la paralelización va trabajar con tuplas, las cuales serán: las filas a y la matriz b completa
    args = [(a[i], b) for i in range(len(a))]

    n_nucleos = 3
    with mp.Pool() as pool:
        # starmap distribuye las tareas y mantiene el orden de las filas
        c = pool.starmap(calcular_fila, args)

    return c




# Main (lo separo en un comentario de las funciones pq sino me confundo xd)
if __name__ == "__main__":
    # Dimensiones de las matrices
    n = 10  # Filas A
    m = 8  # Columnas/Filas A/B
    l = 10  # Columnas B

    # Crear las matrices de forma aleatoria
    a = np.random.randint(0, 9, size=(n, m))
    b = np.random.randint(0, 9, size=(m, l))

    print(f"Matriz A ({n}x{m}):\n{a}\n")
    print(f"Matriz B ({m}x{l}):\n{b}\n")

    # Revisar que las matrices si sean multiplicables
    if (len(a[0]) != len(b)):
        print("Las matrices no se pueden multplicar")
        exit()

    c = multiplicacion_secuencial(a, b)
    print("Resultado secuencial:")
    print(c)

    print("\nResultado Numpy")
    ab = a@b
    print(ab)


    c_mp = multiplicacion_paralela(a,b)
    print("\nResultado mp")
    print(c_mp)