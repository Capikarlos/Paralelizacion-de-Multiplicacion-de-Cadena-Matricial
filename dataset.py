import csv
from pathlib import Path

#Directorio donde se encuentra este archivo
BASE_DIR = Path(__file__).resolve().parent

#Directorio que contiene el dataset
DATA_DIR = BASE_DIR / "data"

#Ruta del archivo original
DATASET_PATH = DATA_DIR / "YearPredictionMSD.txt"


def validar_archivo_dataset():
    #Comprobar el archivo del dataset exista pa ver su ubicacion y tamaño

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset en:\n{DATASET_PATH}\n"
            "Verifica que la carpeta data esté junto a dataset.py."
        )

    # Obtener el tamaño del archivo en MB
    tamano_mb = DATASET_PATH.stat().st_size / (1024 ** 2)

    print("Dataset localizado correctamente.")
    print(f"Ruta: {DATASET_PATH}")
    print(f"Tamaño: {tamano_mb:.2f} MB")

    return DATASET_PATH


def inspeccionar_dataset():
    #Recorrer el archivo para ver bien el dataset

    ruta = validar_archivo_dataset()

    total_filas = 0
    columnas_detectadas = None
    filas_incorrectas = 0
    ejemplos_incorrectos = []
    primeras_filas = []

    #Leer el archivo registro por registro para no cargar todo el dataset en la memoria RAM y me explote aca aaaa
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        lector = csv.reader(archivo)
        for numero_fila, fila in enumerate(lector, start=1):
            #Ignorar registros vacios
            if not fila or not any(valor.strip() for valor in fila):
                continue

            total_filas += 1

            #Identificar el num de columnas de la primera fila
            if columnas_detectadas is None:
                columnas_detectadas = len(fila)

            #Guardar solamente los primeros tres registros
            if len(primeras_filas) < 3:
                primeras_filas.append(fila[:5])

            #YearPredictionMSD debe tener 91 columnas
            if len(fila) != 91:
                filas_incorrectas += 1

                #Conservar como maximo cinco ejemplos
                if len(ejemplos_incorrectos) < 5:
                    ejemplos_incorrectos.append(
                        (numero_fila, len(fila))
                    )

    print("\nEstructura del dataset:")
    print(f"Total de instancias: {total_filas}")
    print(f"Columnas en la primera fila: {columnas_detectadas}")
    print("Columnas esperadas: 91")
    print(f"Filas con columnas incorrectas: {filas_incorrectas}")

    print("\nPrimeros registros (solo los primeros 5 valores):")

    for indice, fila in enumerate(primeras_filas, start=1):
        print(f"Registro {indice}: {fila}")

    if ejemplos_incorrectos:
        print("\nEjemplos de filas con estructura incorrecta:")

        for numero_fila, columnas in ejemplos_incorrectos:
            print(
                f"Fila {numero_fila}: "
                f"{columnas} columnas"
            )

    return total_filas, columnas_detectadas


if __name__ == "__main__":
    inspeccionar_dataset()
