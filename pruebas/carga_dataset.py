import numpy as np
from sklearn.datasets import fetch_california_housing

def obtener_datos_entrenamiento():
    """
    Descarga un dataset masivo de características de viviendas en California.
    Contiene más de 20,000 registros numéricos ideales para regresión lineal.
    """
    datos = fetch_california_housing()
    
    # X son las variables independientes (matrices de características)
    # y es la variable dependiente (precio real a predecir)
    X = datos.data
    y = datos.target.reshape(-1, 1)
    
    print(f"Dataset listo: {X.shape[0]} filas y {X.shape[1]} columnas numéricas.")
    return X, y

if __name__ == "__main__":
    X, y = obtener_datos_entrenamiento()