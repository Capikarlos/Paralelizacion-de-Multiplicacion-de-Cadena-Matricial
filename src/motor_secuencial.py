import numpy as np
from src import fijar_un_hilo_blas
from MultMatricesSec import multiplicacion_secuencial

fijar_un_hilo_blas()


def sigmoide_estable(z: np.ndarray) -> np.ndarray:
    """Función sigmoide con recorte numérico para evitar que se desborde con exp()."""
    z_clip = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


class MotorSecuencial:
    """
    Esto coso corre con solo un núcleo, es decir que funciona de forma secuencial.
    Recorre X por bloques de filas para optimizar la localidad en caché.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, tam_lote: int = 65_536):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.n_muestras, self.n_cols = self.X.shape
        self.tam_lote = tam_lote

    def calcular_gradiente_y_costo(
        self,
        pesos: np.ndarray,
        l2: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        w = np.ascontiguousarray(pesos, dtype=np.float32)
        # Convertimos w (1D) a matriz columna 2D (n_cols x 1) para multiplicacion_secuencial
        w_col = w.reshape(-1, 1).tolist()

        grad_acum = np.zeros(self.n_cols, dtype=np.float64)
        loss_acum = 0.0
        eps = 1e-7

        for inicio in range(0, self.n_muestras, self.tam_lote):
            fin = min(inicio + self.tam_lote, self.n_muestras)
            X_b = self.X[inicio:fin]
            y_b = self.y[inicio:fin]

            # z = X_b @ w pero usando nuestra función
            z_col = multiplicacion_secuencial(X_b.tolist(), w_col)
            z = np.array([fila[0] for fila in z_col], dtype=np.float32)

            p = sigmoide_estable(z)
            error = p - y_b

            # grad_acum += (X_b.T @ error).astype(np.float64) pero usando nuestra función
            error_col = error.reshape(-1, 1).tolist()
            grad_b_col = multiplicacion_secuencial(X_b.T.tolist(), error_col)
            grad_acum += np.array([fila[0] for fila in grad_b_col], dtype=np.float64)

            p_safe = np.clip(p, eps, 1.0 - eps)
            loss_acum -= float(
                np.sum(
                    y_b * np.log(p_safe) + (1.0 - y_b) * np.log(1.0 - p_safe),
                    dtype=np.float64,
                )
            )

        grad = grad_acum / self.n_muestras
        loss = loss_acum / self.n_muestras

        if l2 > 0.0:
            reg = l2 * w.astype(np.float64)
            reg[0] = 0.0
            grad += reg
            loss += 0.5 * l2 * float(np.sum(w[1:].astype(np.float64) ** 2))

        return grad.astype(np.float32), loss

    def cerrar(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cerrar()
        return False