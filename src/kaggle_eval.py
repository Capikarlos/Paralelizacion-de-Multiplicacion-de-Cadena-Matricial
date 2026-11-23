import argparse
import os
import pandas as pd
import numpy as np

from src import fijar_un_hilo_blas

fijar_un_hilo_blas()

from src.dataset import (
    agregar_bias,
    dividir_train_test_estratificado,
    expandir_polinomio_grado2,
    normalizar_zscore,
)
from src.metricas import evaluar_clasificacion, imprimir_reporte_metricas
from src.regresion import RegresionLogisticaParalela


def preparar_datos_kaggle(
    ruta_train: str,
    ruta_test: str,
    col_objetivo: str,
    col_id: str = "id",
    usar_polinomio: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Preprocesa train.csv y test.csv de Kaggle alineando columnas numéricas
    y codificando categóricas con One-Hot Encoding sin fuga de datos.
    """
    df_train = pd.read_csv(ruta_train)
    df_test = pd.read_csv(ruta_test)

    if col_objetivo not in df_train.columns:
        raise ValueError(
            f"La columna objetivo '{col_objetivo}' no existe en {ruta_train}. "
            f"Columnas disponibles: {list(df_train.columns)}"
        )

    ids_test = df_test[col_id] if col_id in df_test.columns else pd.Series(range(len(df_test)))
    y_train = df_train[col_objetivo].to_numpy(dtype=np.float32)

    cols_quitar = [c for c in (col_id, col_objetivo) if c in df_train.columns]
    X_train_df = df_train.drop(columns=cols_quitar)
    X_test_df = df_test.drop(columns=[c for c in (col_id,) if c in df_test.columns])

    # Rellenar faltantes numéricos con mediana de entrenamiento y categóricos con 'desconocido'
    for col in X_train_df.columns:
        if pd.api.types.is_numeric_dtype(X_train_df[col]):
            mediana = X_train_df[col].median()
            X_train_df[col] = X_train_df[col].fillna(mediana)
            if col in X_test_df.columns:
                X_test_df[col] = X_test_df[col].fillna(mediana)
        else:
            X_train_df[col] = X_train_df[col].fillna("desconocido").astype(str)
            if col in X_test_df.columns:
                X_test_df[col] = X_test_df[col].fillna("desconocido").astype(str)

    # One-Hot Encoding conjunto y alineación exacta de columnas
    n_train = len(X_train_df)
    combinado = pd.get_dummies(
        pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True),
        drop_first=True,
        dtype=np.float32,
    )
    X_train = np.ascontiguousarray(combinado.iloc[:n_train].to_numpy(dtype=np.float32))
    X_test = np.ascontiguousarray(combinado.iloc[n_train:].to_numpy(dtype=np.float32))

    if usar_polinomio and X_train.shape[1] <= 35:
        X_train = expandir_polinomio_grado2(X_train)
        X_test = expandir_polinomio_grado2(X_test)

    return X_train, y_train, X_test, ids_test


def ejecutar_flujo_kaggle(args: argparse.Namespace) -> None:
    print(f">>> Cargando datos de Kaggle desde {args.train} y {args.test}...")
    X_full, y_full, X_kaggle, ids_test = preparar_datos_kaggle(
        ruta_train=args.train,
        ruta_test=args.test,
        col_objetivo=args.target,
        col_id=args.col_id,
        usar_polinomio=args.polinomio,
    )

    # 1. Validación local 80/20 para reportar ROC-AUC y Exactitud reales
    X_tr, X_val, y_tr, y_val = dividir_train_test_estratificado(X_full, y_full, test_size=0.2)
    X_tr, X_val, _, _ = normalizar_zscore(X_tr, X_val)
    X_tr = agregar_bias(X_tr)
    X_val = agregar_bias(X_val)

    print(f">>> Validación local | Shape train: {X_tr.shape} | Motor: paralelo_shm ({args.nucleos} procesos)")
    modelo_val = RegresionLogisticaParalela(
        tasa_aprendizaje=args.lr,
        epocas=args.epocas,
        l2=args.l2,
        modo="paralelo_shm",
        num_workers=args.nucleos,
    )
    modelo_val.entrenar(X_tr, y_tr)
    probs_val = modelo_val.predecir_probabilidades(X_val)
    metricas_val = evaluar_clasificacion(y_val, probs_val)
    imprimir_reporte_metricas(metricas_val)

    # 2. Reentrenamiento con el 100% de train.csv para máxima calidad en submission.csv
    X_full_norm, X_kaggle_norm, _, _ = normalizar_zscore(X_full, X_kaggle)
    X_full_norm = agregar_bias(X_full_norm)
    X_kaggle_norm = agregar_bias(X_kaggle_norm)

    print(">>> Entrenando modelo final con 100% de train.csv...")
    modelo_final = RegresionLogisticaParalela(
        tasa_aprendizaje=args.lr,
        epocas=args.epocas,
        l2=args.l2,
        modo="paralelo_shm",
        num_workers=args.nucleos,
    )
    modelo_final.entrenar(X_full_norm, y_full, verbose=False)
    probs_test = modelo_final.predecir_probabilidades(X_kaggle_norm)

    predicciones = (probs_test >= 0.5).astype(int) if args.clases else probs_test
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    submission = pd.DataFrame({args.col_id: ids_test, args.target: predicciones})
    submission.to_csv(args.salida, index=False)
    print(f">>> Archivo listo para subir a Kaggle guardado en: {args.salida} ({len(submission):,} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación Kaggle con motor multiprocessing propio")
    parser.add_argument("--train", default="data/kaggle/train.csv", help="Ruta a train.csv")
    parser.add_argument("--test", default="data/kaggle/test.csv", help="Ruta a test.csv")
    parser.add_argument("--target", required=True, help="Nombre exacto de la columna objetivo en train.csv")
    parser.add_argument("--col-id", default="id", help="Nombre de la columna ID")
    parser.add_argument("--nucleos", type=int, default=None, help="Número de procesos paralelos")
    parser.add_argument("--epocas", type=int, default=40, help="Épocas de entrenamiento")
    parser.add_argument("--lr", type=float, default=0.1, help="Tasa de aprendizaje")
    parser.add_argument("--l2", type=float, default=1e-4, help="Regularización L2")
    parser.add_argument("--polinomio", action="store_true", help="Activar expansión polinómica grado 2")
    parser.add_argument("--clases", action="store_true", help="Exportar 0/1 en vez de probabilidades")
    parser.add_argument("--salida", default="resultados/submission.csv", help="Ruta del CSV de salida")
    ejecutar_flujo_kaggle(parser.parse_args())