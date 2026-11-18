
import time
import numpy as np

from src.motor_secuencial import MotorSecuencial, sigmoide_estable
from src.motor_paralelo import MotorParaleloSHM

class RegresionLogisticaParalela:

    def __init__(
        self,
        tasa_aprendizaje: float = 0.1,
        epocas: int = 40,
        l2: float = 1e-4,
        momentum: float = 0.9,
        modo: str = "paralelo_shm",
        num_workers: int | None = None,
        tam_sublote: int = 32_768,
    ):
        self.lr = tasa_aprendizaje
        self.epocas = epocas
        self.l2 = l2
        self.momentum = momentum
        self.modo = modo
        self.num_workers = num_workers
        self.tam_sublote = tam_sublote

        self.pesos: np.ndarray | None = None
        self.historial_loss: list[float] = []
        self.tiempo_entrenamiento: float = 0.0

    def _crear_motor(self, X: np.ndarray, y: np.ndarray):
        if self.modo == "secuencial":
            return MotorSecuencial(X, y, tam_lote=self.tam_sublote,)

        if self.modo == "paralelo_shm":
            return MotorParaleloSHM(X, y, num_procesos=self.num_workers, tam_sublote=self.tam_sublote,)

        raise ValueError(f"Modo no soportado: '{self.modo}'. ""Usa 'secuencial' o 'paralelo_shm'.")

    def entrenar(self, X: np.ndarray, y: np.ndarray,verbose: bool = True,) -> "RegresionLogisticaParalela":

        n_cols = X.shape[1]
        self.pesos = np.zeros(n_cols, dtype=np.float32)
        velocidad = np.zeros(n_cols, dtype=np.float32)
        self.historial_loss = []

        with self._crear_motor(X, y) as motor:
            t_inicio = time.perf_counter()

            for epoca in range(1, self.epocas + 1):
                grad, loss = motor.calcular_gradiente_y_costo(self.pesos,l2=self.l2,)

                velocidad = self.momentum * velocidad + grad
                self.pesos -= self.lr * velocidad
                self.historial_loss.append(loss)

                if verbose and (epoca == 1 or epoca % 10 == 0 or epoca == self.epocas):
                    print(f"  [{self.modo}] Época {epoca:03d}/{self.epocas} " f"| Log-Loss: {loss:.5f}")

            self.tiempo_entrenamiento = time.perf_counter() - t_inicio

        return self

    def predecir_probabilidades(self, X: np.ndarray, tam_lote: int = 65_536,) -> np.ndarray:

        if self.pesos is None:
            raise RuntimeError("El modelo debe entrenarse antes de predecir.")

        n_filas = X.shape[0]
        probs = np.empty(n_filas, dtype=np.float32)

        for inicio in range(0, n_filas, tam_lote):
            fin = min(inicio + tam_lote, n_filas)
            probs[inicio:fin] = sigmoide_estable(
                X[inicio:fin] @ self.pesos
            )

        return probs

    def predecir(self, X: np.ndarray, umbral: float = 0.5,) -> np.ndarray:

        probs = self.predecir_probabilidades(X)
        return (probs >= umbral).astype(np.int8)
