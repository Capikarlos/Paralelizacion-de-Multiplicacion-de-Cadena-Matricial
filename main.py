import time
from src.dataset import cargar_dataset
from src.regresion import entrenar_regresion_logistica, predecir
from src.metricas import imprimir_reporte_tiempos, imprimir_reporte_modelo

def principal():
    print("Iniciando pipeline de Regresión Logística...")
    
    # Se extraen las características y el año a predecir
    X_np, y_np = cargar_dataset(limite=500)
    
    # Convertir NumPy arrays a listas nativas para compatibilidad con código secuencial
    X = X_np.tolist()
    
    # Binarizar: Si la canción es posterior al 2000, es clase 1. De lo contrario, 0.
    y = [[1] if anio > 2000 else [0] for anio in y_np.tolist()]
    
    EPOCAS = 5
    LR = 0.001
    NUCLEOS = 4
    
    print("\n[ Entrenando modelo SECUENCIAL O(n^3) ]")
    inicio_sec = time.time()
    pesos_sec = entrenar_regresion_logistica(X, y, epochs=EPOCAS, lr=LR, paralelo=False)
    tiempo_sec = time.time() - inicio_sec
    
    print("\n[ Entrenando modelo PARALELO (Tiling) ]")
    inicio_par = time.time()
    pesos_par = entrenar_regresion_logistica(X, y, epochs=EPOCAS, lr=LR, paralelo=True, num_procesos=NUCLEOS)
    tiempo_par = time.time() - inicio_par
    
    print("\nEvaluando modelo...")
    y_pred = predecir(X, pesos_par)
    
    imprimir_reporte_tiempos(tiempo_sec, tiempo_par, NUCLEOS)
    imprimir_reporte_modelo(y, y_pred)

if __name__ == '__main__':
    principal()