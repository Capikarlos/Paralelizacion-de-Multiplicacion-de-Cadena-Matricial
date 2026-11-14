from __future__ import annotations

import os
import time
import numpy as np

from src.motor_secuencial import MotorSecuencial
from src.motor_paralelo import MotorParaleloSHM


def generar_lista_nucleos(max_nucleos: int | None = None) -> list[int]:
    """Construye una progresión de núcleos [1, 2, 4, ..., total] según la PC."""
    total = max_nucleos or (os.cpu_count() or 4)
    lista = []
    p = 1
    while p < total:
        lista.append(p)
        p *= 2
    if not lista or lista[-1] != total:
        lista.append(total)
    return lista


def _medir_motor(motor, n_cols: int, epocas: int, lr: float = 0.1) -> float:
    """Mide el tiempo de entrenamiento durante un número fijo de épocas."""
    w = np.zeros(n_cols, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(epocas):
        grad, _ = motor.calcular_gradiente_y_costo(w)
        w -= lr * grad
    return time.perf_counter() - t0


def benchmark_escalamiento_fuerte(
    X: np.ndarray,
    y: np.ndarray,
    epocas: int = 15,
    lista_nucleos: list[int] | None = None,
    tam_sublote: int = 32_768,
) -> list[dict]:
    """
    Escalamiento Fuerte (Ley de Amdahl):
    Mantiene fijo el tamaño del dataset N y aumenta los procesos de 1 hasta P_max
    usando multiprocessing contra 1 hilo secuencial.
    """
    nucleos = lista_nucleos or generar_lista_nucleos()
    n_cols = X.shape[1]

    with MotorSecuencial(X, y, tam_lote=tam_sublote) as motor_seq:
        t_seq = _medir_motor(motor_seq, n_cols, epocas)

    resultados = [
        {
            "modo": "secuencial",
            "nucleos": 1,
            "tiempo": t_seq,
            "speedup": 1.0,
            "eficiencia": 1.0,
        }
    ]

    for p in nucleos:
        if p == 1:
            continue

        with MotorParaleloSHM(
            X, y, num_procesos=p, tam_sublote=tam_sublote
        ) as motor_shm:
            t_shm = _medir_motor(motor_shm, n_cols, epocas)

        s_shm = t_seq / t_shm
        resultados.append(
            {
                "modo": "paralelo_shm",
                "nucleos": p,
                "tiempo": t_shm,
                "speedup": s_shm,
                "eficiencia": s_shm / p,
            }
        )

    return resultados