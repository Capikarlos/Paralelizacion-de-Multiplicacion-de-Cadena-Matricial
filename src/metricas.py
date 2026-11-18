import numpy as np


def calcular_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray,) -> dict[str, int]:
    # Calcula TP, TN, FP y FN para clasificación binaria.
    yt = np.asarray(y_true, dtype=np.int8)
    yp = np.asarray(y_pred, dtype=np.int8)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calcular_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Calcula el área bajo la curva ROC (ROC-AUC) de forma exacta en NumPy puro manejando empates de probabilidad y usando integración trapezoidal.
    yt = np.asarray(y_true, dtype=np.int8)
    yp = np.asarray(y_prob, dtype=np.float64)

    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    orden = np.argsort(-yp, kind="mergesort")
    y_ord = yt[orden]
    prob_ord = yp[orden]

    cambios = np.where(np.diff(prob_ord))[0]
    indices_umbral = np.r_[cambios, y_ord.size - 1]

    tps = np.cumsum(y_ord, dtype=np.float64)[indices_umbral]
    fps = (indices_umbral + 1).astype(np.float64) - tps

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))


def evaluar_clasificacion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float = 0.5,
) -> dict:
    # Devuelve Accuracy, Precision, Recall, F1, Log-Loss, ROC-AUC y matriz
    yt = np.asarray(y_true, dtype=np.float64)
    yp_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (yp_prob >= umbral).astype(np.int8)

    mc = calcular_matriz_confusion(yt, y_pred)
    tp, tn, fp, fn = mc["TP"], mc["TN"], mc["FP"], mc["FN"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    p_safe = np.clip(yp_prob, 1e-7, 1.0 - 1e-7)
    log_loss = -float(np.mean(yt * np.log(p_safe) + (1.0 - yt) * np.log(1.0 - p_safe)))
    roc_auc = calcular_roc_auc(yt, yp_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "log_loss": log_loss,
        "matriz_confusion": mc,
    }


def calcular_metricas_hpc(tiempo_secuencial: float, tiempo_paralelo: float, num_nucleos: int,) -> dict[str, float]:
    #Calcula Speedup (S), Eficiencia (E) y porcentaje de reducción de tiempo
    speedup = tiempo_secuencial / tiempo_paralelo if tiempo_paralelo > 0 else 0.0
    eficiencia = speedup / num_nucleos if num_nucleos > 0 else 0.0
    reduccion_pct = ((1.0 - (tiempo_paralelo / tiempo_secuencial)) * 100.0
        if tiempo_secuencial > 0
        else 0.0)
    return {
        "tiempo_secuencial": tiempo_secuencial,
        "tiempo_paralelo": tiempo_paralelo,
        "nucleos": num_nucleos,
        "speedup": speedup,
        "eficiencia": eficiencia,
        "reduccion_tiempo_pct": reduccion_pct,
    }


def imprimir_reporte_metricas(metricas_ml: dict, metricas_hpc: dict | None = None) -> None:
    #Muestra en consola un resumen limpio listo para reportes o diapositivas.
    mc = metricas_ml["matriz_confusion"]
    print("\n MÉTRICAS DE CLASIFICACIÓN ")
    print(f"  Exactitud (Accuracy) : {metricas_ml['accuracy'] * 100:6.2f}%")
    print(f"  Precisión            : {metricas_ml['precision'] * 100:6.2f}%")
    print(f"  Recall               : {metricas_ml['recall'] * 100:6.2f}%")
    print(f"  F1-Score             : {metricas_ml['f1_score'] * 100:6.2f}%")
    print(f"  ROC-AUC              : {metricas_ml['roc_auc']:.4f}")
    print(f"  Log-Loss             : {metricas_ml['log_loss']:.4f}")
    print(
        f"  Matriz de Confusión  : TP={mc['TP']} | FP={mc['FP']} | "
        f"TN={mc['TN']} | FN={mc['FN']}"
    )

    if metricas_hpc is not None:
        print("\n MÉTRICAS DE CÓMPUTO PARALELO")
        print(f"  Tiempo Secuencial    : {metricas_hpc['tiempo_secuencial']:.3f} s")
        print(
            f"  Tiempo Paralelo ({metricas_hpc['nucleos']}c) : "
            f"{metricas_hpc['tiempo_paralelo']:.3f} s"
        )
        print(f"  Speedup (S)          : {metricas_hpc['speedup']:.2f}x")
        print(f"  Eficiencia (E)       : {metricas_hpc['eficiencia'] * 100:.2f}%")
        print(f"  Reducción de tiempo  : {metricas_hpc['reduccion_tiempo_pct']:.2f}%")
    print("---------------\n")