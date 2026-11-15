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

def benchmark_escalamiento_debil(
    X: np.ndarray,
    y: np.ndarray,
    filas_base: int = 150_000,
    epocas: int = 15,
    lista_nucleos: list[int] | None = None,
    tam_sublote: int = 32_768,
) -> list[dict]:
    """
    Escalamiento Débil (Ley de Gustafson):
    Aumenta el volumen de datos proporcionalmente a los núcleos (N = p * filas_base)
    usando MotorParaleloSHM para medir estabilidad de tiempo y speedup escalado.
    """
    nucleos = lista_nucleos or generar_lista_nucleos()
    n_cols = X.shape[1]
    resultados = []
    t_base = None

    for p in nucleos:
        n_actual = p * filas_base
        if n_actual > X.shape[0]:
            break

        X_sub = X[:n_actual]
        y_sub = y[:n_actual]

        if p == 1:
            with MotorSecuencial(X_sub, y_sub, tam_lote=tam_sublote) as motor:
                t_p = _medir_motor(motor, n_cols, epocas)
            t_base = t_p
        else:
            with MotorParaleloSHM(
                X_sub, y_sub, num_procesos=p, tam_sublote=tam_sublote
            ) as motor:
                t_p = _medir_motor(motor, n_cols, epocas)

        eficiencia_debil = t_base / t_p if t_base else 1.0
        resultados.append(
            {
                "nucleos": p,
                "filas": n_actual,
                "tiempo": t_p,
                "speedup_gustafson": p * eficiencia_debil,
                "eficiencia_debil": eficiencia_debil,
            }
        )

    return resultados


def verificar_equivalencia_motores(
    X: np.ndarray,
    y: np.ndarray,
    num_procesos: int = 4,
    tolerancia: float = 1e-4,
) -> float:
    """
    Comprueba matemáticamente que MotorSecuencial y MotorParaleloSHM producen
    el mismo gradiente dentro de una tolerancia numérica para asegurar que los
    resultados sean similares o iguales.
    """
    rng = np.random.default_rng(42)
    w = rng.standard_normal(X.shape[1]).astype(np.float32) * 0.1

    with MotorSecuencial(X, y) as m_seq:
        g_seq, l_seq = m_seq.calcular_gradiente_y_costo(w)

    with MotorParaleloSHM(X, y, num_procesos=num_procesos) as m_par:
        g_par, l_par = m_par.calcular_gradiente_y_costo(w)

    diff_grad = float(np.max(np.abs(g_seq - g_par)))
    diff_loss = abs(l_seq - l_par)
    if diff_grad > tolerancia or diff_loss > tolerancia:
        raise AssertionError(
            f"Discrepancia entre motores: diff_grad={diff_grad:.2e}, diff_loss={diff_loss:.2e}"
        )
    return diff_grad