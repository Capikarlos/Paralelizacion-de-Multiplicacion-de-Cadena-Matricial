import math
from paralelo import multiplicacion_paralela_bloques
from MultMatricesSec import multiplicacion_secuencial

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
    #Entrenar el modelo de regresion log mediante el descenso de gradiente
    #X: matriz de caracteristicas, y: etiquetas binarias con forma (num_muestras, 1).
    #epochs: num de epocas de entrenamiento, lr: tasa de aprendizaje, paralelo: indica si se utiliza la multiplicacion paralela.
    #num_procesos: cantidad de procesos para la multiplicacion paralela.

    num_muestras = len(X)
    num_caracteristicas = len(X[0])

    #Inicializar los pesos en cero
    pesos = [[0.0] for _ in range(num_caracteristicas)]

    #Entrenar durante el num de epocas indicado
    for epoca in range(epochs):
        #Calcular X * pesos
        if paralelo: 
            z = multiplicacion_paralela_bloques(X, pesos, num_procesos=num_procesos)
        else:
            z = multiplicacion_secuencial(X, pesos)

        #Convertir los resultados en probabilidades
        h = [[sigmoide(val[0])] for val in z]

        #Calcular el error entre las predicciones y las etiquetas
        error = [[h[i][0] - y[i][0]]
            for i in range(num_muestras)]

        #Transponer X para calcular el gradiente
        X_T = transponer_matriz(X)

        #Calcular X^T * error
        if paralelo:
            gradiente_bruto = multiplicacion_paralela_bloques( X_T, error, num_procesos=num_procesos)
        else:
            gradiente_bruto = multiplicacion_secuencial(X_T, error)

        #Actualizar los pesos usando el gradiente promedio
        for i in range(num_caracteristicas):
            gradiente_promedio = (gradiente_bruto[i][0] / num_muestras)
            pesos[i][0] -= lr * gradiente_promedio

    return pesos