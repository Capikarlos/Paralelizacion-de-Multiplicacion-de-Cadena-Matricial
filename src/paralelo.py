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
