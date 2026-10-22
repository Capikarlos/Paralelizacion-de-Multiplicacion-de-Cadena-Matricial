# src/metricas.py

def calcular_speedup(t_secuencial, t_paralelo):
    # Relación entre el tiempo de ejecución de un solo procesador y múltiples procesadores
    if t_paralelo == 0:
        return 0
    return t_secuencial / t_paralelo


def calcular_eficiencia(speedup, num_procesos):
    # Porcentaje de utilización real de los núcleos asignados
    if num_procesos == 0:
        return 0
    return speedup / num_procesos


def metricas_clasificacion(y_real, y_pred):
    # Inicialización de la matriz de confusión
    vp = 0  # Verdaderos Positivos
    vn = 0  # Verdaderos Negativos
    fp = 0  # Falsos Positivos
    fn = 0  # Falsos Negativos

    for real, pred in zip(y_real, y_pred):
        # Manejo de listas anidadas por si los datos vienen como [[1], [0]]
        val_real = real[0] if isinstance(real, list) else real
        val_pred = pred[0] if isinstance(pred, list) else pred

        if val_real == 1 and val_pred == 1:
            vp += 1
        elif val_real == 0 and val_pred == 0:
            vn += 1
        elif val_real == 0 and val_pred == 1:
            fp += 1
        elif val_real == 1 and val_pred == 0:
            fn += 1

    total = len(y_real)
    
    # Cálculo de métricas manejando divisiones por cero
    exactitud = (vp + vn) / total if total > 0 else 0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return exactitud, precision, recall, f1


def imprimir_reporte_tiempos(t_sec, t_par, num_procesos):
    speedup = calcular_speedup(t_sec, t_par)
    eficiencia = calcular_eficiencia(speedup, num_procesos)
    print(f"Tiempo Secuencial: {t_sec:.4f} s")
    print(f"Tiempo Paralelo: {t_par:.4f} s")
    print(f"Speedup: {speedup:.2f}")
    print(f"Eficiencia: {eficiencia:.4f}\n")


def imprimir_reporte_modelo(y_real, y_pred):
    exactitud, precision, recall, f1 = metricas_clasificacion(y_real, y_pred)
    print("\nEvaluación")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
