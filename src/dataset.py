import os
import numpy as np
import pandas as pd


def cargar_dataset_csv(
    ruta_csv: str,
    max_filas: int | None = None,
    col_objetivo: int | str = 0,
    tiene_encabezado: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un dataset tabular binario masivo en float32 para reducir
    a la mitad el consumo de RAM y acelerar el ancho de banda de memoria.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_csv}")

    header = 0 if tiene_encabezado else None
    df = pd.read_csv(
        ruta_csv,
        header=header,
        nrows=max_filas,
        dtype=np.float32,
        engine="c",
    )

    # Eliminar filas con valores faltantes si existen
    if df.isna().any().any():
        df = df.dropna()

    if isinstance(col_objetivo, str):
        y = df[col_objetivo].to_numpy(dtype=np.float32)
        X = df.drop(columns=[col_objetivo]).to_numpy(dtype=np.float32)
    else:
        y = df.iloc[:, col_objetivo].to_numpy(dtype=np.float32)
        X = df.drop(df.columns[col_objetivo], axis=1).to_numpy(dtype=np.float32)

    # Asegurar arreglo contiguo en memoria (C-order) para acceso rápido por filas
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)
    return X, y


def generar_dataset_sintetico(
    n_muestras: int = 1_000_000,
    n_caracteristicas: int = 18,
    ruido: float = 0.35,
    semilla: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset binario masivo con relaciones lineales y cuadráticas
    para benchmarks rápidos de Amdahl/Gustafson sin depender de archivos externos.
    """
    rng = np.random.default_rng(semilla)
    X = rng.standard_normal((n_muestras, n_caracteristicas), dtype=np.float32)

    pesos_reales = rng.uniform(-1.5, 1.5, size=n_caracteristicas).astype(np.float32)
    # Incluir término lineal + interacción cuadrática para simular física de SUSY/HIGGS
    logits = X @ pesos_reales + 0.4 * (X[:, 0] ** 2 - X[:, 1] ** 2)
    logits += rng.normal(0.0, ruido, size=n_muestras).astype(np.float32)

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y = (probs >= 0.5).astype(np.float32)

    return np.ascontiguousarray(X), np.ascontiguousarray(y)