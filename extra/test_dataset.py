#Librerias q se van a usar aa
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import src.dataset as dataset

#Probar el dataset de musica
class TestDataset(unittest.TestCase):
    def setUp(self):
        #Crear una carpeta temporal para las pruebas
        self.directorio_temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.directorio_temporal.cleanup)

        self.directorio = Path(self.directorio_temporal.name)

        self.ruta_dataset = self.directorio / "YearPredictionMSD.txt"
        self.ruta_cache = self.directorio / "YearPredictionMSD.npy"

        #Crear cuatro instancias con 91 columnas
        self.datos_prueba = np.zeros((4, 91), dtype=np.float32)

        #Primera columna es el año que queremos predecir
        self.datos_prueba[:, 0] = [2000, 2001, 2002, 2003]

        #Columnas restantes son las variables predictoras
        self.datos_prueba[:, 1:] = np.arange(4 * 90, dtype=np.float32).reshape(4, 90)

        #Guardar el dataset de prueba
        np.savetxt(self.ruta_dataset, self.datos_prueba, delimiter=",", fmt="%.6f")

        # Usar las rutas temporales en lugar del dataset real
        parche_dataset = patch.object(dataset, "DATASET_PATH", self.ruta_dataset)
        parche_cache = patch.object(dataset, "CACHE_PATH", self.ruta_cache)

        parche_dataset.start()
        parche_cache.start()

        self.addCleanup(parche_dataset.stop)
        self.addCleanup(parche_cache.stop)

    #Cargar todas las instancias del test
    def test_instancias(self):
        #Comprobar la carga completa
        X, y = dataset.cargar_dataset()

        self.assertEqual(X.shape, (4, 90))
        self.assertEqual(y.shape, (4,))

        self.assertEqual(X.dtype, np.float32)

        np.testing.assert_array_equal(y, self.datos_prueba[:, 0])

    #Cargar una muestra
    def test_muestra(self):
        #Comprobar el limite
        X, y = dataset.cargar_dataset(limite=2)

        self.assertEqual(X.shape, (2, 90))
        self.assertEqual(y.shape, (2,))

        np.testing.assert_array_equal(y, self.datos_prueba[:2, 0])

    def test_limites(self):
        #Verificar que se rechacen limites incorrectos
        limites_invalidos = [0, -1, 1.5, True]

        for limite in limites_invalidos:
            with self.subTest(limite=limite):
                with self.assertRaises(ValueError):
                    dataset.cargar_dataset(limite=limite)

    #Crear y reusar el cache
    def test_cache(self):
        #La primera carga debe crear la cache
        X, y = dataset.cargar_dataset(limite=2)

        self.assertTrue(self.ruta_cache.is_file())

        #Asegurar que la cache tenga una fecha posterior al archivo original, para simular actualizacion
        fecha_cache = self.ruta_dataset.stat().st_mtime + 10

        os.utime(self.ruta_cache, (fecha_cache, fecha_cache))

        #La segunda carga reusa el cache
        salida = io.StringIO()

        with contextlib.redirect_stdout(salida):
            X2, y2 = dataset.cargar_dataset(limite=2)

        self.assertIn("Utilizando la caché binaria existente", salida.getvalue())

        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)

    def test_inspeccion(self):
        #Verificar el conteo de filas y columnas
        with contextlib.redirect_stdout(io.StringIO()):
            filas, columnas = dataset.inspeccionar_dataset()

        self.assertEqual(filas, 4)
        self.assertEqual(columnas, 91)

    def test_archivo_inexistente(self):
        # Comprobar el comportamiento si no existe el archivo
        self.ruta_dataset.unlink()

        with self.assertRaises(FileNotFoundError):
            dataset.validar_archivo_dataset()

    #Rechazar archivo con columnas incorrectas
    def test_columnas_incorrectas(self):
        # Crear un archivo con 90 columnas en vez de 91
        datos_incorrectos = np.zeros((2, 90), dtype=np.float32)

        np.savetxt(self.ruta_dataset, datos_incorrectos, delimiter=",", fmt="%.6f")

        with self.assertRaisesRegex(ValueError, "91 columnas"):
            dataset.cargar_dataset()

if __name__ == "__main__":
    unittest.main(verbosity=2)