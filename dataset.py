
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


if __name__ == "__main__":
    validar_archivo_dataset()
