import atexit
import multiprocessing as mp
import numpy as np

from src import fijar_un_hilo_blas
from src.memoria_compartida import (
    GestorMemoriaCompartida,
    conectar_arreglo_compartido,
)
from src.motor_secuencial import sigmoide_estable

_SHM_HANDLES = []
_X_VISTA = None
_Y_VISTA = None


def _cerrar_handles_worker() -> None:
    for shm in _SHM_HANDLES:
        try:
            shm.close()
        except Exception:
            pass
    _SHM_HANDLES.clear()


def _inicializar_worker(meta_X: dict, meta_y: dict) -> None:
    global _X_VISTA, _Y_VISTA
    fijar_un_hilo_blas()
    shm_X, _X_VISTA = conectar_arreglo_compartido(meta_X)
    shm_y, _Y_VISTA = conectar_arreglo_compartido(meta_y)
    _SHM_HANDLES.extend([shm_X, shm_y])
    atexit.register(_cerrar_handles_worker)


def calcular_rangos_balanceados(n_filas: int, num_partes: int) -> list[tuple[int, int]]:
    """Reparte las filas con diferencia máxima de 1 fila entre núcleos."""
    base, residuo = divmod(n_filas, num_partes)
    rangos = []
    inicio = 0
    for idx in range(num_partes):
        tam = base + (1 if idx < residuo else 0)
        if tam > 0:
            rangos.append((inicio, inicio + tam))
        inicio += tam
    return rangos


def _worker_gradiente_optimizado(args: tuple) -> tuple[np.ndarray, float]:
    """Procesa su partición en micro-lotes amigables con caché L2/L3."""
    pesos, inicio, fin, tam_sublote = args
    n_cols = _X_VISTA.shape[1]
    grad_local = np.zeros(n_cols, dtype=np.float64)
    loss_local = 0.0
    eps = 1e-7

    for pos in range(inicio, fin, tam_sublote):
        limite = min(pos + tam_sublote, fin)
        X_b = _X_VISTA[pos:limite]
        y_b = _Y_VISTA[pos:limite]

        z = X_b @ pesos
        p = sigmoide_estable(z)
        error = p - y_b

        grad_local += (X_b.T @ error).astype(np.float64)
        p_safe = np.clip(p, eps, 1.0 - eps)
        loss_local -= float(
            np.sum(
                y_b * np.log(p_safe) + (1.0 - y_b) * np.log(1.0 - p_safe),
                dtype=np.float64,
            )
        )

    return grad_local, loss_local


class MotorParaleloSHM:
    """
    Motor multinúcleo ahora más molón:
    - Zero-Copy (no crea copias del dataset, todo apunta a la misma dirección de memoria
      para que tarde menos y la memoria no colapse) con multiprocessing.shared_memory.
    - Particionado balanceado + micro-lotes locales para minimizar perdida de memoria cache.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_procesos: int | None = None,
        tam_sublote: int = 32_768,
    ):
        fijar_un_hilo_blas()
        self.n_muestras, self.n_cols = X.shape
        self.num_procesos = num_procesos or mp.cpu_count()
        self.tam_sublote = tam_sublote

        self.gestor = GestorMemoriaCompartida()
        meta_X = self.gestor.registrar_arreglo("X", X)
        meta_y = self.gestor.registrar_arreglo("y", y)

        self.rangos = calcular_rangos_balanceados(self.n_muestras, self.num_procesos)
        self.pool = mp.Pool(
            processes=len(self.rangos),
            initializer=_inicializar_worker,
            initargs=(meta_X, meta_y),
        )

    def calcular_gradiente_y_costo(
        self,
        pesos: np.ndarray,
        l2: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        w = np.ascontiguousarray(pesos, dtype=np.float32)
        tareas = [
            (w, inicio, fin, self.tam_sublote)
            for inicio, fin in self.rangos
        ]
        resultados = self.pool.map(_worker_gradiente_optimizado, tareas, chunksize=1)

        grad_acum = np.zeros(self.n_cols, dtype=np.float64)
        loss_acum = 0.0
        for grad_p, loss_p in resultados:
            grad_acum += grad_p
            loss_acum += loss_p

        grad = grad_acum / self.n_muestras
        loss = loss_acum / self.n_muestras

        if l2 > 0.0:
            reg = l2 * w.astype(np.float64)
            reg[0] = 0.0
            grad += reg
            loss += 0.5 * l2 * float(np.sum(w[1:].astype(np.float64) ** 2))

        return grad.astype(np.float32), loss

    def cerrar(self) -> None:
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        if getattr(self, "gestor", None) is not None:
            self.gestor.liberar()
            self.gestor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cerrar()
        return False