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


def dividir_train_test_estratificado(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    semilla: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en entrenamiento y prueba conservando exactamente
    la proporción original de clases 0 y 1 (estratificado).
    """
    rng = np.random.default_rng(semilla)
    idx_0 = np.flatnonzero(y == 0)
    idx_1 = np.flatnonzero(y == 1)

    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    corte_0 = int(len(idx_0) * (1.0 - test_size))
    corte_1 = int(len(idx_1) * (1.0 - test_size))

    idx_train = np.concatenate([idx_0[:corte_0], idx_1[:corte_1]])
    idx_test = np.concatenate([idx_0[corte_0:], idx_1[corte_1:]])

    rng.shuffle(idx_train)
    rng.shuffle(idx_test)

    return (
        np.ascontiguousarray(X[idx_train]),
        np.ascontiguousarray(X[idx_test]),
        np.ascontiguousarray(y[idx_train]),
        np.ascontiguousarray(y[idx_test]),
    )


def normalizar_zscore(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Aplica estandarización Z-Score: (X - media) / desv_est.
    Calcula media y desviación estándar en float64 únicamente sobre X_train
    para evitar fuga de datos (data leakage) y errores de precisión numérica.
    """
    media = np.mean(X_train, axis=0, dtype=np.float64).astype(np.float32)
    desv = np.std(X_train, axis=0, dtype=np.float64).astype(np.float32)
    # Proteger contra columnas constantes (evita división por cero)
    desv[desv < 1e-8] = 1.0

    X_train_norm = np.ascontiguousarray((X_train - media) / desv, dtype=np.float32)
    X_test_norm = None
    if X_test is not None:
        X_test_norm = np.ascontiguousarray((X_test - media) / desv, dtype=np.float32)

    return X_train_norm, X_test_norm, media, desv


def expandir_polinomio_grado2(X: np.ndarray) -> np.ndarray:
    """
    Genera características de grado 2 (cuadrados e interacciones x_i * x_j).
    Por ejemplo, en SUSY pasa de 18 atributos a 189 atributos, aumentando
    tanto la capacidad predictiva lineal como la intensidad aritmética.
    Preasigna la matriz en float32 para evitar picos innecesarios de RAM.
    """
    n_filas, n_cols = X.shape
    n_interacciones = (n_cols * (n_cols + 1)) // 2
    n_total = n_cols + n_interacciones

    X_poly = np.empty((n_filas, n_total), dtype=np.float32)
    X_poly[:, :n_cols] = X

    col_actual = n_cols
    for i in range(n_cols):
        xi = X[:, i : i + 1]
        n_bloque = n_cols - i
        np.multiply(xi, X[:, i:], out=X_poly[:, col_actual : col_actual + n_bloque])
        col_actual += n_bloque

    return X_poly


def agregar_bias(X: np.ndarray) -> np.ndarray:
    """
    Inserta una columna de unos al inicio de X para representar el término
    independiente (intercepto / bias) sin necesidad de normalizarlo.
    """
    unos = np.ones((X.shape[0], 1), dtype=np.float32)
    return np.ascontiguousarray(np.hstack((unos, X)), dtype=np.float32)


def preparar_pipeline_datos(
    ruta_csv: str | None = None,
    max_filas: int | None = None,
    n_sintetico: int = 1_000_000,
    n_features_sintetico: int = 18,
    usar_polinomio: bool = True,
    test_size: float = 0.2,
    col_objetivo: int | str = 0,
    tiene_encabezado: bool = False,
    semilla: int = 42,
) -> dict:
    """
    Orquesta el preprocesamiento completo de extremo a extremo:
    1. Carga (CSV real o sintético)
    2. Split estratificado 80/20
    3. Expansión polinómica grado 2 (opcional)
    4. Normalización Z-Score ajustada solo en entrenamiento
    5. Adición de columna de bias
    """
    if ruta_csv:
        X, y = cargar_dataset_csv(
            ruta_csv=ruta_csv,
            max_filas=max_filas,
            col_objetivo=col_objetivo,
            tiene_encabezado=tiene_encabezado,
        )
    else:
        X, y = generar_dataset_sintetico(
            n_muestras=n_sintetico,
            n_caracteristicas=n_features_sintetico,
            semilla=semilla,
        )

    X_train, X_test, y_train, y_test = dividir_train_test_estratificado(
        X, y, test_size=test_size, semilla=semilla
    )

    if usar_polinomio:
        X_train = expandir_polinomio_grado2(X_train)
        X_test = expandir_polinomio_grado2(X_test)

    X_train, X_test, media, desv = normalizar_zscore(X_train, X_test)

    X_train = agregar_bias(X_train)
    X_test = agregar_bias(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "media": media,
        "desv": desv,
    }