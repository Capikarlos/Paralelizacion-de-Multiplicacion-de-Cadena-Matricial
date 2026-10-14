import multiprocessing as mp
import math

matriz_A = None
matriz_B = None
def iniciar_worker(A_compartida, B_compartida):
    """Inicializa las variables globales en cada proceso hijo."""
    global matriz_A, matriz_B
    matriz_A = A_compartida
    matriz_B = B_compartida

def calcular_bloque_resultado(args):
    """
    Realiza la multiplicación utilizando la técnica de Bloques. Procesa sub-cuadrículas para que los datos quepan en la memoria caché L1/L2 del procesador.
    """
    i_start, i_end, block_size = args
    global matriz_A, matriz_B
    
    M = len(matriz_A[0]) # Columnas de A e Filas de B
    L = len(matriz_B[0]) # Columnas de B
    
    # Inicializar con ceros solo la porción de la matriz C que le toca a este núcleo
    resultado_parcial = [[0] * L for _ in range(i_end - i_start)]
    # Algoritmo de Multiplicación por Bloques
    for j in range(0, L, block_size):
        for k in range(0, M, block_size):
            
            # Multiplicación tradicional
            for i_local, i_global in enumerate(range(i_start, i_end)):
                for jj in range(j, min(j + block_size, L)):
                    suma_temp = 0
                    for kk in range(k, min(k + block_size, M)):

                        suma_temp += matriz_A[i_global][kk] * matriz_B[kk][jj]
                    resultado_parcial[i_local][jj] += suma_temp
                    
    # Retornamos el índice inicial para poder ordenar los pedazos después
    return (i_start, resultado_parcial)


def multiplicacion_paralela_bloques(A, B, num_procesos=None, block_size=32):
    """
    Función principal que divide el trabajo y lanza el Pool de procesos.
    """
    if num_procesos is None:
        num_procesos = mp.cpu_count()
        
    N = len(A)
    
    # Dividir las filas equitativamente entre el número de núcleos
    filas_por_proceso = math.ceil(N / num_procesos)
    tareas = []
    
    for i in range(0, N, filas_por_proceso):
        i_end = min(i + filas_por_proceso, N)
        tareas.append((i, i_end, block_size))
        
    # Lanzar multiprocesamiento
    with mp.Pool(processes=num_procesos, initializer=iniciar_worker, initargs=(A, B)) as pool:
        resultados = pool.map(calcular_bloque_resultado, tareas)
        
    # El Pool puede devolver los resultados en desorden, los ordenamoos por el i_start
    resultados.sort(key=lambda x: x[0])
    
    # Ensamblar la matriz final C uniendo las filas calculadas
    C = []
    for _, bloque in resultados:
        C.extend(bloque)
        
    return C