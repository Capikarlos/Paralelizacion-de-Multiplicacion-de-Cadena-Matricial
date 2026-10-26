import math

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


