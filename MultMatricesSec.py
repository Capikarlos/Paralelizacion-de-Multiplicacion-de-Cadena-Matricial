import numpy as np

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


if __name__ == "__main__":
    a = np.array([[3, 4, 5],
                  [2, 4, 2],
                  [1, 5, 5]])

    b = np.array([[4, 2, 4],
                  [0, 1, 5],
                  [3, 6, 8]])

    # Revisar que las matrices si sean multiplicables
    if (len(a[0]) != len(b)):
        print("Las matrices no se pueden multplicar")
        exit()

    c = multiplicacion_secuencial(a, b)
    print(c)

    ab = a@b
    print(ab)