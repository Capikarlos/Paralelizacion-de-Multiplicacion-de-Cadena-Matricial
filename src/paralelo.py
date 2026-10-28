import multiprocessing as mp
import math

def calcular_bloque_resultado(args):
    i_start, A_chunk, B = args
    
    M = len(A_chunk[0]) # Columnas de A / Filas de B
    L = len(B[0])       # Columnas de B
    
    # Transponer B para leer filas contiguas en lugar de columnas (Acceso contiguo en memoria)
    B_T = [[B[k][j] for k in range(M)] for j in range(L)]
    
    resultado_parcial = []
    
    # Uso de iteradores directos (ejecución nativa acelerada en CPython)
    for fila_A in A_chunk:
        fila_res = []
        for fila_B in B_T:
            suma = 0.0
            for k in range(M):
                suma += fila_A[k] * fila_B[k]
            fila_res.append(suma)
        resultado_parcial.append(fila_res)
        
    return (i_start, resultado_parcial)


def multiplicacion_paralela_bloques(A, B, num_procesos=None, block_size=None):
    if num_procesos is None:
        num_procesos = mp.cpu_count()
        
    N = len(A)
    
    # Dividir el número total de filas entre el número de procesos
    filas_por_proceso = math.ceil(N / num_procesos)
    tareas = []
    
    for i in range(0, N, filas_por_proceso):
        i_end = min(i + filas_por_proceso, N)
        A_chunk = A[i:i_end]
        tareas.append((i, A_chunk, B))
        
    # Lanzar multiprocesamiento
    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(calcular_bloque_resultado, tareas)
        
    resultados.sort(key=lambda x: x[0])
    
    # Ensamblar la matriz final
    C = []
    for _, bloque in resultados:
        C.extend(bloque)
        
    return C