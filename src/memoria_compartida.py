from multiprocessing import shared_memory
import numpy as np


class GestorMemoriaCompartida:
    """
    Administra arreglos NumPy en memoria compartida
    Permite que múltiples procesos hijos lean las matrices X e y directamente
    desde la misma región de RAM sin serialización (pickling) ni duplicación.
    """

    def __init__(self):
        self._bloques: dict[str, shared_memory.SharedMemory] = {}
        self.metadatos: dict[str, dict] = {}

    def registrar_arreglo(self, clave: str, arreglo: np.ndarray) -> dict:
        """
        Reserva memoria compartida para un arreglo contiguo y copia sus bytes
        una sola vez antes de lanzar el Pool de procesos.
        """
        arr = np.ascontiguousarray(arreglo)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        vista = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(vista, arr)

        meta = {
            "name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
        self._bloques[clave] = shm
        self.metadatos[clave] = meta
        return meta

    def liberar(self) -> None:
        """Cierra y desvincula todos los bloques para evitar fugas en /dev/shm."""
        for shm in self._bloques.values():
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._bloques.clear()
        self.metadatos.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.liberar()
        return False


def conectar_arreglo_compartido(
    meta: dict,
) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """
    Usado dentro de cada proceso trabajador (worker) para adjuntarse a un bloque
    de memoria existente con cero copias. El trabajador debe cerrar el objeto
    SharedMemory al finalizar (sin hacer unlink).
    """
    shm = shared_memory.SharedMemory(name=meta["name"])
    vista = np.ndarray(
        meta["shape"],
        dtype=np.dtype(meta["dtype"]),
        buffer=shm.buf,
    )
    return shm, vista