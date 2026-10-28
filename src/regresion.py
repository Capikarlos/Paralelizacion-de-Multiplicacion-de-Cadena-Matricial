import math
import time
from .paralelo import multiplicacion_paralela_bloques
from .MultMatricesSec import multiplicacion_secuencial

def transponer_matriz(matriz):
    #Transpone una matriz intercambiando filas por columnas
    return [
        [matriz[j][i] for j in range(len(matriz))]
        for i in range(len(matriz[0]))
    ]

def sigmoide(z):
    #Calcula la funcion sig para convertir el valor a una probabilidad d 0 o 1
    if z < -250:
        return 0.0
    if z > 250:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def entrenar_regresion_logistica(X, y, epochs=5, lr=0.001, paralelo=False, num_procesos=4):
    # Asegurar que trabajamos con listas nativas (por si el dataset.py manda numpy arrays por la caché)
    if hasattr(X, "tolist"): X = X.tolist()
    if hasattr(y, "tolist"): y = y.tolist()

    num_muestras = len(X)
    num_caracteristicas = len(X[0])

    #Asegurar que 'y' sea una matriz columna de tipo [[y1], [y2], ...]
    if not isinstance(y[0], list):
        y = [[val] for val in y]

    #Inicializar los pesos en cero
    pesos = [[0.0] for _ in range(num_caracteristicas)]

    print("\n[INFO] Preparando datos: calculando la matriz transpuesta...")
    inicio_transpuesta = time.time()
    
    #OPTIMIZACION: Transponer X una sola vez antes de empezar las epocas
    X_T = transponer_matriz(X)
    
    print(f"[INFO] Matriz transpuesta lista en {time.time() - inicio_transpuesta:.2f} segundos.")
    
    tipo_ejecucion = "PARALELO" if paralelo else "SECUENCIAL"
    print(f"\n[INFO] Iniciando entrenamiento {tipo_ejecucion} por {epochs} epocas...")

    #Entrenar durante el num de epocas indicado
    for epoca in range(epochs):
        inicio_epoca = time.time()
        print(f"  -> Iniciando Epoca {epoca + 1}/{epochs}...")
        
        #Calcular X * pesos
        if paralelo: 
            z = multiplicacion_paralela_bloques(X, pesos, num_procesos=num_procesos)
        else:
            z = multiplicacion_secuencial(X, pesos)

        #Convertir los resultados en probabilidades
        h = [[sigmoide(val[0])] for val in z]

        #Calcular el error entre las predicciones y las etiquetas
        error = [[h[i][0] - y[i][0]] for i in range(num_muestras)]

        #Calcular X^T * error (usando la transpuesta ya calculada)
        if paralelo:
            gradiente_bruto = multiplicacion_paralela_bloques(X_T, error, num_procesos=num_procesos)
        else:
            gradiente_bruto = multiplicacion_secuencial(X_T, error)

        #Actualizar los pesos usando el gradiente promedio
        for i in range(num_caracteristicas):
            gradiente_promedio = (gradiente_bruto[i][0] / num_muestras)
            pesos[i][0] -= lr * gradiente_promedio
            
        tiempo_epoca = time.time() - inicio_epoca
        print(f"     [OK] Epoca {epoca + 1} completada en {tiempo_epoca:.2f} segundos.")

    return pesos

def predecir(X, pesos):
    #Predice la clase de cada instancia usando los pesos obtenidos y retorna las predicciones
    if hasattr(X, "tolist"): X = X.tolist()
    
    print("\n[INFO] Generando predicciones finales")
    
    #Calcular X * pesos
    z = multiplicacion_secuencial(X, pesos)

    predicciones = []
    #Convertir las probabilidades en clases
    for val in z:
        probabilidad = sigmoide(val[0])
        clase = 1 if probabilidad >= 0.5 else 0
        predicciones.append([clase])

    return predicciones