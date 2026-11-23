
import argparse
import os

from src import fijar_un_hilo_blas

#Fijar 1 hilo BLAS antes de importar NumPy o cualquier submódulo
fijar_un_hilo_blas()

from src.dataset import preparar_pipeline_datos
from src.escalamiento import benchmark_escalamiento_debil, benchmark_escalamiento_fuerte
from src.graficas import graficar_convergencia, graf_escalamientod, graficar_escalamiento_fuerte, graficar_matriz_confusion
from src.metricas import calcular_metricas_hpc, evaluar_clasificacion, imprimir_reporte_metricas
from src.regresion import RegresionLogisticaParalela


#Configuracion de argumentos
def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paralelización de Operaciones Matriciales para Regresión Logística Masiva (v2.0)")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Ruta al CSV/CSV.GZ real (ej. data/SUSY.csv.gz). Si se omite usa datos sintéticos.")

    parser.add_argument("--max-filas", type=int, default=1_000_000,
                        help="Máximo de filas a cargar del CSV o a generar en modo sintético.")

    parser.add_argument("--sin-polinomio", action="store_true",
                        help="Desactivar expansión polinómica grado 2.")

    parser.add_argument("--epocas", type=int, default=30,
                        help="Número de épocas de entrenamiento.")

    parser.add_argument("--lr", type=float, default=0.1,
                        help="Tasa de aprendizaje.")

    parser.add_argument("--l2", type=float, default=1e-4,
                        help="Factor de regularización L2.")

    parser.add_argument("--nucleos", type=int, default=os.cpu_count() or 4,
                        help="Número de núcleos para el entrenamiento paralelo principal.")

    parser.add_argument("--benchmark", action="store_true",
                        help="Ejecutar curvas completas de Amdahl y Gustafson y generar gráficas.")

    return parser.parse_args()


#Funcion principal
def main() -> None:
    #Obtener argumentos
    args = parsear_argumentos()
    usar_poli = not args.sin_polinomio

    #Preparacion del dataset
    print(">>> [1/4] Preparando pipeline de datos (float32, Z-Score, Bias), aguantenme un rato")

    datos = preparar_pipeline_datos(ruta_csv=args.dataset,
                                    max_filas=args.max_filas,
                                    n_sintetico=args.max_filas,
                                    usar_polinomio=usar_poli)

    X_train, X_test = datos["X_train"], datos["X_test"]
    y_train, y_test = datos["y_train"], datos["y_test"]

    print(f"    Train: {X_train.shape} ({X_train.nbytes / 1e6:.1f} MB) | "
          f"Test: {X_test.shape}")

    #Entrenamiento secuencial
    print("\n>>> [2/4] Entrenando línea base SECUENCIAL (1 hilo estricto), ya casi")

    modelo_seq = RegresionLogisticaParalela(tasa_aprendizaje=args.lr,
                                          epocas=args.epocas,
                                          l2=args.l2,
                                          modo="secuencial")

    modelo_seq.entrenar(X_train, y_train)

    #Entrenamiento paralelo
    print(f"\n>>> [3/4] Entrenando motor PARALELO Zero-Copy "
          f"(multiprocessing.shared_memory, {args.nucleos} procesos), aguantenmen")

    modelo_par = RegresionLogisticaParalela(tasa_aprendizaje=args.lr,
                                          epocas=args.epocas,
                                          l2=args.l2,
                                          modo="paralelo_shm",
                                          num_workers=args.nucleos)

    modelo_par.entrenar(X_train, y_train)

    #Evaluacion con los datos de prueba
    probs_test = modelo_par.predecir_probabilidades(X_test)
    metricas_ml = evaluar_clasificacion(y_test, probs_test)

    #Calculo de metricas de computo paralelo
    metricas_hpc = calcular_metricas_hpc(tiempo_secuencial=modelo_seq.tiempo_entrenamiento,
                                        tiempo_paralelo=modelo_par.tiempo_entrenamiento,
                                        num_nucleos=args.nucleos)

    #Impresion de resultados
    imprimir_reporte_metricas(metricas_ml, metricas_hpc)

    #Generacion de graficas
    graficar_convergencia(modelo_par.historial_loss)
    graficar_matriz_confusion(metricas_ml["matriz_confusion"])

    #Pruebas de escalamiento
    if args.benchmark:

        print(">>> [4/4] Ejecutando benchmarks de Escalamiento Fuerte (Amdahl) y Débil (Gustafson), ola capi")

        epocas_bench = min(15, args.epocas)

        #Escalamiento fuerte (Ley de Amdahl)
        res_fuerte = benchmark_escalamiento_fuerte(X_train, y_train, epocas=epocas_bench)

        print("\n--- Escalamiento Fuerte (Ley de Amdahl) ---")

        for r in res_fuerte:
            print(f"  {r['modo']:15s} | P={r['nucleos']:2d} | "
                  f"T={r['tiempo']:6.2f}s | S={r['speedup']:5.2f}x | "
                  f"E={r['eficiencia'] * 100:5.1f}%")

        graficar_escalamiento_fuerte(res_fuerte)

        #Escalamiento debil (Ley de Gustafson)
        filas_base = max(50_000, X_train.shape[0] // max(args.nucleos, 1))

        res_debil = benchmark_escalamiento_debil(X_train, y_train,
                                                filas_base=filas_base,
                                                epocas=epocas_bench)

        print("\n--- Escalamiento Débil (Ley de Gustafson) ---")

        for r in res_debil:
            print(f"  P={r['nucleos']:2d} | Filas={r['filas']:9,d} | "
                  f"T={r['tiempo']:6.2f}s | S_Gustafson={r['speedup_gustafson']:5.2f}x")

        graf_escalamientod(res_debil)

        print("\n>>> Gráficas guardadas en la carpeta 'resultados/'")

    else:
        print(">>> Gráficas de convergencia y confusión guardadas en 'resultados/'.")


if __name__ == "__main__":
    main()