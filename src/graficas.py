import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _asegurar_directorio(ruta_dir: str) -> None:
    os.makedirs(ruta_dir, exist_ok=True)


def graficar_escalamiento_fuerte(
    resultados: list[dict],
    ruta_salida: str = "resultados/escalamiento_fuerte_amdahl.png",
) -> None:
    """Genera gráfica doble de Speedup y Eficiencia vs número de núcleos."""
    _asegurar_directorio(os.path.dirname(ruta_salida))

    seq = next(r for r in resultados if r["modo"] == "secuencial")
    shm = sorted(
        [r for r in resultados if r["modo"] == "paralelo_shm"],
        key=lambda x: x["nucleos"],
    )

    nucleos_shm = [1] + [r["nucleos"] for r in shm]
    speedup_shm = [1.0] + [r["speedup"] for r in shm]
    efic_shm = [100.0] + [r["eficiencia"] * 100.0 for r in shm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax1.plot(nucleos_shm, nucleos_shm, "--", color="gray", label="Speedup Ideal (S = P)")
    ax1.plot(nucleos_shm, speedup_shm, "o-", linewidth=2, color="#1f77b4", label="Multiprocessing SHM (Zero-Copy)")
    ax1.set_title(f"Escalamiento Fuerte — Ley de Amdahl\n(Base secuencial: {seq['tiempo']:.2f} s)")
    ax1.set_xlabel("Número de núcleos (P)")
    ax1.set_ylabel("Speedup (S)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(100.0, linestyle="--", color="gray", label="Eficiencia Ideal (100%)")
    ax2.plot(nucleos_shm, efic_shm, "o-", linewidth=2, color="#ff7f0e", label="Multiprocessing SHM")
    ax2.set_title("Eficiencia Paralela por Núcleo")
    ax2.set_xlabel("Número de núcleos (P)")
    ax2.set_ylabel("Eficiencia (%)")
    ax2.set_ylim(0, 115)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close(fig)


def graf_escalamientod(resultados: list[dict], ruta_salida: str = "resultados/escalamiento_debil_gustafson.png",) -> None:
    # Genera gráfica de Speedup escalado y estabilidad de tiempo (Ley de Gustafson)
    if not resultados:
        return
    _asegurar_directorio(os.path.dirname(ruta_salida))

    nucleos = [r["nucleos"] for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]
    speedup_g = [r["speedup_gustafson"] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax1.plot(nucleos, nucleos, "--", color="gray", label="Escalado Lineal Ideal")
    ax1.plot(nucleos, speedup_g, "o-", color="#1f77b4", linewidth=2, label="Speedup Escalado (Gustafson)")
    ax1.set_title("Escalamiento Débil — Ley de Gustafson")
    ax1.set_xlabel("Número de núcleos (P) con carga proporcional")
    ax1.set_ylabel("Speedup Escalado")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.bar([str(p) for p in nucleos], tiempos, color="#2ca02c", alpha=0.85)
    ax2.set_title("Estabilidad de Tiempo con Carga Proporcional")
    ax2.set_xlabel("Número de núcleos (P)")
    ax2.set_ylabel("Tiempo de ejecución (s)")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close(fig)


def graficar_convergencia(historial_loss: list[float], ruta_salida: str = "resultados/convergencia_logloss.png",) -> None:
    # Grafica la caída del Binary Cross-Entropy (Log-Loss) por epoca
    _asegurar_directorio(os.path.dirname(ruta_salida))

    epocas = np.arange(1, len(historial_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(epocas, historial_loss, "o-", linewidth=2, markersize=4, color="#d62728")
    ax.set_title("Convergencia del Modelo — Binary Cross-Entropy (Log-Loss)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Log-Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close(fig)


def graficar_matriz_confusion(mc: dict[str, int], ruta_salida: str = "resultados/matriz_confusion.png",) -> None:
    # Genera mapa de calor limpio de la matriz de confusión.# 
    _asegurar_directorio(os.path.dirname(ruta_salida))

    matriz = np.array([[mc["TN"], mc["FP"]], [mc["FN"], mc["TP"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    im = ax.imshow(matriz, cmap="Blues")
    plt.colorbar(im, ax=ax)

    etiquetas = ["Clase 0", "Clase 1"]
    ax.set_xticks([0, 1], labels=etiquetas)
    ax.set_yticks([0, 1], labels=etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    ax.set_title("Matriz de Confusión (Conjunto de Prueba)")

    umbral_color = matriz.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if matriz[i, j] > umbral_color else "black"
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close(fig)