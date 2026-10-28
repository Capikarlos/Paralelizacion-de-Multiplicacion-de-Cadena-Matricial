import math

# Multiplicación secuencial con cota O(n³)
def multiplicacion_secuencial(a, b):
    c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(b[0])):
            suma = 0

            for k in range(len(a[0])):
                suma += a[i][k] * b[k][j]

            c[i][j] = suma

    return c