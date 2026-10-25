import numpy as np
import multiprocessing as mp
import time # Añadido para medir tiempos

# Variable global para que los procesos compartan la matriz B sin duplicarla en la RAM
matriz_b_global = None

def inicializar_pool(b_compartida):
    global matriz_b_global
    matriz_b_global = b_compartida

# Prueba secuencial original
def multiplicacion_secuencial(a, b):
    c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            suma = 0
            for k in range(len(a[0])):
                suma += a[i][k] * b[k][j]
            c[i][j] = suma
    return c

# Función paralela optimizada para RAM (usa la matriz B global)
def calcular_fila_optimizada(fila_a):
    global matriz_b_global
    fila_resultado = [0 for _ in range(len(matriz_b_global[0]))]
    
    for j in range(len(matriz_b_global[0])):
        suma = 0
        for k in range(len(fila_a)):
            suma += fila_a[k] * matriz_b_global[k][j]
        fila_resultado[j] = suma
        
    return fila_resultado

def multiplicacion_paralela(a, b):
    # Ya no duplicamos 'b' en los argumentos, solo mandamos las filas de 'a'
    n_nucleos = mp.cpu_count() # Usa los núcleos disponibles
    
    # Pasamos 'b' como variable inicializadora para compartir memoria
    with mp.Pool(processes=n_nucleos, initializer=inicializar_pool, initargs=(b,)) as pool:
        c = pool.map(calcular_fila_optimizada, a)
    return c

def imprimir_matriz(matriz):
    for fila in matriz:
        print(fila)

if __name__ == "__main__":
    n, m, l = 100, 80, 100  # Aumenté el tamaño para que se note la diferencia de tiempo

    a_np = np.random.randint(0, 9, size=(n, m))
    b_np = np.random.randint(0, 9, size=(m, l))
    a = a_np.tolist()
    b = b_np.tolist()

    if (len(a[0]) != len(b)):
        print("Las matrices no se pueden multiplicar")
        exit()

    print(f"Multiplicando matrices de ({n}x{m}) y ({m}x{l})...\n")

    # --- Medición Secuencial ---
    inicio_sec = time.time()
    c = multiplicacion_secuencial(a, b)
    fin_sec = time.time()
    print(f"Tiempo secuencial: {fin_sec - inicio_sec:.5f} segundos")

    # --- Medición Paralela ---
    inicio_par = time.time()
    c_mp = multiplicacion_paralela(a, b)
    fin_par = time.time()
    print(f"Tiempo en paralelo: {fin_par - inicio_par:.5f} segundos")